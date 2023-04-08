from typing import List, Union, Dict, Any
import torch
import pickle
import numpy as np
import concurrent.futures

from tqdm import tqdm
from pathlib import Path
from transformers import BertModel, BertTokenizer

import utils
import logging
# ToDo; reduce the redundancy in the class methods

logger = logging


class Embedder:
    def __init__(self, tokenizer: BertTokenizer,
                 bert: BertModel,
                 IDF_dict: Dict[str, float],
                 embedding_dir: Path,
                 layers_to_use: List[int] = [12],
                 layer_combination: str = "avg",
                 idf_threshold: float = 1.5,
                 idf_weight_factor: float = 1.0,
                 not_found_idf_value: float = 0.5):
        """
        Object that provides functinoality to tokenize some input text, as well as computes the
        (potentially IDF weighted) embeddings.

        :param tokenizer:    Pretrained BertTokenizer.
        :param bert:    Pretrained BERT model - from the same BERT model as the tokenizer.
        :param IDF_dict:    Dictionary holding the pre-computed IDF weights for token indices.
        :param layers_to_use:   A list with layer indices to combine. Default is [12], meaning only the output
                                of the last layer is used.
        :param layer_combination:   In case of multiple layers, determine how to combine them. Defaults to `"avg"`,
                                    choice between (`"avg"`, `"sum"`, `"max"`).
        :param idf_threshold:    Minimum IDF weight value for the token to be included during weighting.
        :param idf_weight_factor:    Factor for multiplying IDF weights to stress/reduce the difference between high/low.
        :param not_found_idf_value:    Value for tokens that do not have a corresponding IDF weight(typically low).
        """
        self.tokenizer = tokenizer
        self.bert = bert
        self.IDF_dict = IDF_dict
        self.layers_to_use = layers_to_use
        self.layer_combination = layer_combination
        self.idf_threshold = idf_threshold
        self.idf_weight_factor = idf_weight_factor
        self.not_found_idf_value = not_found_idf_value

        self.embedding_prefix = "embeddings"
        self.embedding_dir = embedding_dir
        self.span_and_embedding_pairs = []
        self.standardised_embedding_data = []
        self.unique_spans = []
        self.new_embedding_data = []
        self.new_unique_spans = []
        self.emb_mean_fp = embedding_dir.joinpath("standardisation_mean.pkl")
        self.emb_std_fp = embedding_dir.joinpath("standardisation_std.pkl")
        self.emb_mean = "None"  # initialise to specific value, later will hold an array (without truth value blabla)
        self.emb_std = "None"

    def prepare_tokens(self, text: str) -> (List[str], List[int]):
        """
        Helper function to tokenize and ensure the same output is provided by the BertTokenizer
        (and BertWordPieceTokenizer ~ deprecated).

        :param tokenizer:    A pretrained BertTokenizer.
        :param text:    Input text.
        :return tokenized_text: List of string representations of tokens for the input text.
        :return indices:    List of token index representations of tokens for the input text.
        """
        # if type(self.tokenizer) == BertWordPieceTokenizer:
        #     encoded = self.tokenizer.encode(text)
        #     tokenized_text = encoded.tokens
        #     indices = encoded.ids
        #     return tokenized_text, indices
        if type(self.tokenizer) == BertTokenizer:
            tokenized_text = self.tokenizer.tokenize("[CLS] " + text + " [SEP]")
            indices = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            if len(tokenized_text) > 512:
                # Any tokens where the index is higher than 512 (max BERT input) will be omitted
                # todo; make this adjustable for different pretrained embedding models like a longformer model
                print("Number of input tokens too long! Truncating input to 512 tokens...")
                tokenized_text, indices = tokenized_text[:512], indices[:512]

            return tokenized_text, indices
        else:
            print("You'll need to change to a BertTokenizer")
            return None  # need to raise appropriate error or quit running nicely

    def get_idf_weights_for_indices(self,
                                    tokenized_text: List[str],
                                    indices: List[int]) -> np.array:
        """
        Helper function to get the relevant IDF weights given tokenized input. Note: if an IDF value is not found, then
        the weight for that token is set to `self.not_found_idf_value`.

        :param tokenized_text:  List of tokens, currently not used. Previously used to visualise which part of the
                                input text is filtered through IDF-based filtering, and which weights are assigned.
        :param indices: Token-indices that are also used to index into the corresponding IDF values, used to retrieve
                        the corresponding IDF weights computed for a domain corpus.
        :return sw_weights: An np.array of the same length as the parameter `indices`, holding IDF weights for the
                            subword unit tokens.
        """
        sw_weights = np.ones(len(indices))
        visualised_text = ''
        for sw_id, sw in enumerate(indices):
            try:
                # set index of phrase representation to corresponding IDF value
                sw_weights[sw_id] = self.IDF_dict[str(sw)]
                visualised_text += ' ' + tokenized_text[sw_id]
            except (KeyError, ValueError) as e:

                # No IDF value found, which value do we set it to?
                sw_weights[sw_id] = self.not_found_idf_value
                visualised_text += ' ' + tokenized_text[sw_id] + '\u0336'

        return sw_weights

    def embed_text(self, text: str) -> List[torch.tensor]:
        """
        :param text:    Text to be tokenized. Expecting BERT-tokenizer, so max input of 512 tokens.

        :return weighted_term_embeddings:    The embeddings for the each of the tokens in the sentence, potentially with  subword
                                    IDF weights applied (multiplied by IDF value, mediated by `self.idf_weight_factor`).
        """
        tokenized_text, indices = self.prepare_tokens(text)

        IDF_weights = self.get_idf_weights_for_indices(tokenized_text, indices)

        segments_ids = [1] * len(indices)
        tokens_tensor = torch.tensor([indices])
        segments_tensors = torch.tensor([segments_ids])
        embedding = self.bert(tokens_tensor, segments_tensors)

        try:
            hidden_states = embedding[2]
        except IndexError:
            raise IndexError("Make sure to output hidden states; BertModel.from_pretrained(some-bert-model, "
                             "output_hidden_states=True)")

        # Group by token vectors
        token_embeddings = torch.stack(hidden_states, dim=0).squeeze(dim=1).permute(1, 0, 2)
        weighted_term_embeddings = []
        for token, IDF_w in zip(token_embeddings, IDF_weights):
            # todo -- consider running more experiments with different BERT layers and combinations
            # Combine the vectors from selected bert layers --> currently default to last layer only [12]
            if self.layer_combination == "sum":
                combined_vec = torch.sum(token.index_select(0, torch.tensor(self.layers_to_use)), dim=0)
            elif self.layer_combination == "max":
                combined_vec = torch.max(token.index_select(0, torch.tensor(self.layers_to_use)), dim=0)
            else:
                # default to self.layer_combination == "avg":
                combined_vec = torch.mean(token.index_select(0, torch.tensor(self.layers_to_use)), dim=0)

            # Weight the vector by IDF value
            if IDF_w > self.idf_threshold and self.not_found_idf_value > 0:  # avoid multiplying by 0
                if self.idf_weight_factor > 0:
                    weighted_term_embeddings.append(combined_vec * (self.idf_weight_factor * IDF_w))
                else:
                    weighted_term_embeddings.append(combined_vec)

        return weighted_term_embeddings

    def combine_token_embeddings(self, embeddings: List[torch.tensor]) -> torch.tensor:
        """
        Calls the embedding function for a span, stacking the embeddings for each token and computing the mean over
        the token-length dimension.

        :param embeddings:   List of tensors for each of the tokens in a given span.
        :return embedding:   Single embedding of the span, as an average over the constituent token embeddings.
        """
        weighted_token_embeddings = torch.stack(embeddings)
        detached_embeddings = weighted_token_embeddings.detach().numpy()
        # Standardise PLM representation
        if type(self.emb_mean) == str and self.emb_mean_fp.exists():
            self.emb_mean = pickle.load(open(self.emb_mean_fp, 'rb'))
            self.emb_std = pickle.load(open(self.emb_std_fp, 'rb'))

        standardised_embeddings = (detached_embeddings - self.emb_mean) / self.emb_std
        return np.mean(standardised_embeddings, axis=0)   # average over the tokens for now. old >> torch.mean(x, dim=0)

    def embed_a_span(self, span: str) -> torch.tensor:
        """
        Use to simply embed a span, BEFORE the mean and std of all spans are computed
        """
        embeddings = self.embed_text(span)
        try:
            return span, torch.stack(embeddings).detach().numpy().squeeze()
        except RuntimeError:
            # can happen if the tensor for the span is empty somehow
            logger.info(f"Empty tensor! Not sure why, but will drop the span: {span}")

    def embed_and_normalise_span(self, span: str) -> torch.tensor:
        """
        Use to embed new spans, AFTER the mean and std of all spans are computed.
        Returns a tuple: (span, normalised_embedding)
        """
        embeddings = self.embed_text(span)
        try:
            return span, self.combine_token_embeddings(embeddings)
        except RuntimeError:
            # can happen if the tensor for the span is empty or causes issues somehow
            print(f"Empty tensor! Not sure why, but will drop the span: {span}")
            return None, None

    def normalise_embeddings(self):
        """
        Creates a single file with all embeddings, in the meantime standardising the embeddings to improve the
        representations (Timkey & van Schijndel, 2021).
        """
        unique_spans, unique_embeddings = zip(*self.span_and_embedding_pairs)
        self.unique_spans = unique_spans
        unique_spans_fp = self.embedding_dir.joinpath("unique_spans.pkl")
        if not unique_spans_fp.exists():
            with open(unique_spans_fp, 'wb') as f:
                pickle.dump(unique_spans, f)

        standardised_classifier_data_fp = self.embedding_dir.joinpath("standardised_embeddings.pkl")
        if not standardised_classifier_data_fp.exists():
            print(
              f"Normalising and combining computed/existing {len(unique_spans)} embeddings from files into single file"
            )
            # we average over the token embeddings in a term
            unique_clustering_data = np.stack(
                [np.mean(e, axis=0) if len(e.shape) > 1 else e for e in unique_embeddings])

            # standardise the unique clustering data, as suggested by https://github.com/wtimkey/rogue-dimensions
            self.emb_mean = unique_clustering_data.mean(axis=0)
            self.emb_std = unique_clustering_data.std(axis=0)
            pickle.dump(self.emb_mean, open(self.embedding_dir.joinpath("standardisation_mean.pkl"), 'wb'))
            pickle.dump(self.emb_std, open(self.embedding_dir.joinpath("standardisation_std.pkl"), 'wb'))

            # Store the span and normalised embedding pairs
            # Note: could remove all the embedding files (keeping them though)
            normalised_embeddings = (unique_clustering_data - self.emb_mean) / self.emb_std
            self.standardised_embedding_data = [(s, e) for s, e in zip(self.unique_spans, normalised_embeddings)]
            pickle.dump(self.standardised_embedding_data, open(standardised_classifier_data_fp, 'wb'))
        else:
            self.standardised_embedding_data = pickle.load(open(standardised_classifier_data_fp, 'rb'))
            print(f"Loaded previously computed normalised embedding files.")

    def embed_fore_and_background_terms(self,
                                        max_num_cpu_threads: int = 4,
                                        subset_size: int = 1000,
                                        list_of_terms: List[str] = None,
                                        prefix: str = 'embeddings'
                                        ):
        """
        This is split into subsets so we don't overload memory (adjust values if needed). Once computed, all
        of the embeddings will be stored normalised. Normalised embeddings and corresponding spans can be accessed
         through `self.standardised_embedding_data`.

        :param max_num_cpu_threads:     Number of concurrent threads.
        :param subset_size: Number of terms embedded at a time, this is the memory bottleneck.
        :param prefix:  Prefix to the name given to the embeddings.
        :param list_of_terms:   A list of spans/terms that you'd like to embed. If None, then the foreground and
                                background terms from the classifier directory will be embedded.
        """
        term_subsets = utils.split_list(list_of_terms, subset_size)
        if prefix != 'embeddings':
            self.embedding_prefix = prefix

        embedding_files = [f for f in self.embedding_dir.glob(f'{prefix}*.pkl')]

        if len(embedding_files) == len(term_subsets):
            for e in embedding_files:
                self.span_and_embedding_pairs += pickle.load(open(e, 'rb'))
        else:
            print(f"Preparing embeddings for {len(list_of_terms)} spans, in groups of: {subset_size}")
            subset_idx = 0  # iterator index outside of tqdm
            for subset in tqdm(term_subsets):
                subset_embeddings = []
                subset_file_name = self.embedding_dir.joinpath(f"{prefix}_part_{subset_idx}.pkl")
                subset_idx += 1
                if subset_file_name.exists():
                    continue

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_cpu_threads) as executor:
                    futures = [executor.submit(self.embed_a_span, subset[idx]) for idx in range(len(subset))]

                subset_embeddings += [f.result() for f in futures if f.result()]

                with open(subset_file_name, 'wb') as f:
                    pickle.dump(subset_embeddings, f)

            # Once all embeddings are created; combine them in span_and_embedding_pairs
            embedding_files = [f for f in self.embedding_dir.glob(f'{prefix}*.pkl')]
            for e in embedding_files:
                self.span_and_embedding_pairs += pickle.load(open(e, 'rb'))

        self.normalise_embeddings()

    def embed_large_number_of_new_terms(self,
                                        max_num_cpu_threads: int = 4,
                                        subset_size: int = 1000,
                                        prefix: str = 'new_terms',
                                        list_of_terms: List[str] = None
                                        ):
        """
        This function is used to embed new spans that are found during KG creation.

        :param max_num_cpu_threads:     Number of concurrent threads.
        :param subset_size: Number of terms embedded at a time, this is the memory bottleneck.
        :param prefix:  Prefix to the name given to the embeddings.
        :param list_of_terms:   A list of spans/terms that you'd like to embed. If None, then the foreground and
                                background terms from the classifier directory will be embedded
        """
        # Checks which of the embeddings for the clustering cluster_data already exist, so they can be re-used
        unique_old_spans, _ = zip(*self.standardised_embedding_data)
        unique_new_spans = [t for t in list_of_terms if t not in unique_old_spans]

        term_subsets = utils.split_list(unique_new_spans, subset_size)
        embedding_files = [f for f in self.embedding_dir.glob(f'{prefix}*.pkl')]
        new_span_and_embedding_pairs = []
        if len(embedding_files) == len(term_subsets):
            for e in embedding_files:
                new_span_and_embedding_pairs += pickle.load(open(e, 'rb'))
        else:
            print(f"Preparing embeddings for {len(unique_new_spans)} spans, in groups of: {subset_size}")
            subset_idx = 0  # iterator index outside of tqdm
            for subset in tqdm(term_subsets):
                subset_embeddings = []
                subset_file_name = self.embedding_dir.joinpath(f"{prefix}_part_{subset_idx}.pkl")
                subset_idx += 1
                if subset_file_name.exists():
                    # already computed previously
                    continue

                # NOTE: the spans are embedded AND normalised in this method.
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_cpu_threads) as executor:
                    futures = [executor.submit(self.embed_and_normalise_span, subset[idx]) for idx in
                               range(len(subset))]

                subset_embeddings += [f.result() for f in futures if f.result()]

                with open(subset_file_name, 'wb') as f:
                    pickle.dump(subset_embeddings, f)

            # Once all embeddings are created; combine them in span_and_embedding_pairs
            embedding_files = [f for f in self.embedding_dir.glob(f'{prefix}*.pkl')]
            for e in embedding_files:
                new_span_and_embedding_pairs += pickle.load(open(e, 'rb'))

        self.new_embedding_data += new_span_and_embedding_pairs
