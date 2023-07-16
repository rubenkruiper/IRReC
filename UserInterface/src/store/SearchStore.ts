import { defineStore } from "pinia";

export const SearchStore = defineStore("searchStore", {
  state: () => ({
    query: null as String,
    predictions: {
      combined_prediction: [],
      query: "",
    },
    loaded: true,
    currentPage: 1,
  }),
  getters: {
    // getStatus: (state) => {
    //   if (state.section1.sectionComplete) return 2;

    //   return 1;
    // },
    getCurrentPage: (state) => {
      return state.currentPage;
    },
    getLoaded: (state) => {
      return state.loaded;
    },
    getPredictions: (state) => {
      return state.predictions;
    },
  },
  actions: {
    setCurrentPage(pageNumber) {
      this.currentPage = pageNumber;
    },
    setLoaded(loaded) {
      this.loaded = loaded;
    },
    setPredictions(predictions) {
      this.loaded = true;

      this.predictions = predictions;
    },
    async getResults(query) {
      return await fetch(`http://localhost:8000/q/${query}`, {
        method: "POST",
        headers: {
          Accept: "application/json",
        },
        //mode: "no-cors",
        //body: payload,
      })
        .then((response) => {
          if (response.status === 200) {
            return response.json();
          } else if (response.status === 400) {
            return {
              success: false,
              message: "Data was malformed",
            };
          }
        })
        .catch((error) => {
          return {
            success: false,
            message: "The server has not responded",
          };
        });
    },
  },
});
