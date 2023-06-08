
# precision, recall, F1
def precision_at_k(bool_list, k=9):
    """ How many items in top k results were relevant """
    return sum(bool_list[:k]) / len(bool_list[:k])


def recall_at_k(bool_list, k=9):
    """ How many relevant items were retrieved within the top k  """
    if sum(bool_list):
        return sum(bool_list[:k]) / sum(bool_list) # + sum([not b for b in bool_list[:k]]) # what are false negs here?
    else:
        return 0
    

def F1_at_k(bool_list, k=9):
    precision = precision_at_k(bool_list, k)
    recall = recall_at_k(bool_list, k)
    if precision and recall:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0


# Mean Reciprocal Rank 
def MRR_query(bool_list):
    """ MRR for a single query """
    try:
        first_index = bool_list.index(True) + 1
        return 1 / first_index
    except:
        return 0


# Average Precision
def AP_query(bool_list):
    """ Average precision for single query """
    AP_sum = 0
    num_positive = sum(bool_list)
    if not num_positive:
        return 0
    
    for rank, annotation in enumerate(bool_list):
        if annotation:
            weight = 1
        else: 
            weight = 0
        prec_k = precision_at_k(bool_list, rank+ 1)
        AP_sum += prec_k * weight
    
    return AP_sum / num_positive
        
