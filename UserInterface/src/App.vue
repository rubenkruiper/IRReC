<script lang="ts">

import Search from './components/Search.vue'
import Results from './components/Results.vue'
import { ref, computed, defineComponent } from 'vue'
import { SearchStore } from '@/store/SearchStore'
import Pagination from './components/Pagination.vue'


export default defineComponent({
  setup() {

    const store = SearchStore();
    const predictions = computed(() => { return store.getPredictions });
    const loaded = computed(() => { return store.getLoaded });

    const currentPage = computed(() => { return store.getCurrentPage });

    function onPageChange(page) {
      store.setCurrentPage(page);
    }

    return {
      predictions,
      loaded,
      store,
      currentPage,
      onPageChange
    }
  },
  components: {
    Search,
    Results,
    Pagination,

  }

})

</script>


<template>
  <div class="h-fit overflow-hidden">
    <div class=" ">
      <div class="flex justify-center my-12">
        <div class="text-center text-5xl">
          IRReC demonstrator
        </div>
      </div>

      <div class="py-4">
        <Search></Search>
      </div>

      <div class="py-4 mb-8">
        <div v-for=" (prediction, index) in predictions.combined_prediction">

          <Results :predictionId="index" :data="prediction">
          </Results>

        </div>
      </div>
      <div :class="{ 'opacity-0': loaded === false || predictions.combined_prediction.length <= 0 }">
        <pagination :totalPages="Math.floor(predictions.combined_prediction.length / 10)" :perPage="10"
          @pagechanged="onPageChange" />
      </div>
    </div>
  </div>

</template>

