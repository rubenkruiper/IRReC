
<script lang="ts">

import { defineComponent, ref, computed, onMounted, watch } from 'vue'
import { SearchStore } from '@/store/SearchStore'

export default defineComponent({
  props: {
    predictionId: Number,
    data: Object,

  },
  setup(props) {
    const predictionId = props.predictionId
    const data = props.data
    const isLoaded = ref(false);
    const shouldShowRef = ref(false);

    let upperLimitOfPages = computed(() => { return Math.ceil((store.getCurrentPage)) * 10 })
    let lowerLimitOfPages = computed(() => { return (Math.ceil((store.getCurrentPage)) * 10) - 10 })

    const documentName = computed(() => {
      return props.data[1].doc_title.split("#")[0]
    })

    const documentDescriptor = computed(() => {
      return props.data[1].doc_title.split("#")[1]
    })

    const store = SearchStore();
    const loaded = computed(() => { return store.getLoaded });

    const shouldShow = computed(() => {
      if (predictionId >= lowerLimitOfPages.value && predictionId < upperLimitOfPages.value) {

        setTimeout(() => {
          shouldShowRef.value = true;
        }, 500)

        return true
      }
      else {
        setTimeout(() => {
          shouldShowRef.value = false;
        }, 500)
        return false
      }

    });

    function splitPage(content) {
      return content.split("##")[1]
    }

    watch(shouldShow, async (before, after) => {
      console.log("state change", after)
      if (after === false) {
        // setTimeout(() => {
        //   shouldShowRef.value = true;
        // }, 500)
      }
      else {
        // setTimeout(() => {
        //   shouldShowRef.value = false;
        // }, 500)
      }
    }
    )

    watch(loaded, async (before, after) => {
      if (after === false) {
        setTimeout(() => {
          isLoaded.value = true;
        }, 500)
      }
      else {
        setTimeout(() => {
          isLoaded.value = false;
        }, 500)
      }
    }
    )

    onMounted(() => {
      setTimeout(() => {
        isLoaded.value = true;
      }, 500)

    })

    return {
      data,
      isLoaded,
      loaded,
      documentName,
      documentDescriptor,
      shouldShowRef,
      splitPage


    }
  }
})
</script>

<template>

  <div class="flex justify-center m-6 transition-opacity duration-500 ease-in-out"
    :class="{ 'h-0 m-0 opacity-0 w-2': isLoaded === false || shouldShowRef === false }">

    <div class=" p-6 rounded-lg shadow-lg bg-white w-10/12  ">
      <div class="md:flex block mb-4 mx-2">
        <h5
          class="text-gray-900 hover:text-blue-600 transition-all duration-500 ease-in-out text-lg leading-tight font-medium flex truncate px-1  flex-grow min-w-fit mr-8">
          <a class="">{{ documentName }}</a>
        </h5>
        <div
          class="text-sm  tracking-wider leading-tight mt-1  md:mb-0 h-8 flex flex-row-reverse	truncate  align-right px-1">
          <p class="truncate  inline-block align-middle">{{
              documentDescriptor
          }}</p>

        </div>
        <div class="flex-row-reverse md:flex-none md:mx-4 italic text-sm text-gray-700 text-right px-1 mt-1">
          {{ data[1].times_retrieved }} Hits
        </div>
      </div>
      <div class=" ">


        <div class=" mt-4 lg:mt-0 mb-6 text-gray-700 bg-gray-50 rounded-lg shadow-inner  flex-grow flex-col-reverse"
          :class="{ 'h-0 m-0 opacity-0 w-2': isLoaded === false || shouldShowRef === false }">
          <div v-for=" (prediction, index) in data[1].contents">
            <div v-if="index < 3" class="p-6  flex "
              :class="{ 'h-0 m-0 opacity-0 w-2': isLoaded === false || shouldShowRef === false }">
              <div class="p-2">{{ index + 1 }}.</div>
              <div class="p-2 flex "> <span class="mx-1">p </span><span> {{ splitPage(prediction.id) }} </span></div>
              <div class="p-2 text-sm break-words min-w-full pr-16 truncate-words "> {{ prediction.text }}</div>
            </div>
          </div>
        </div>
        <div class=" m-4 my-auto flex-none lg:flex-grow ">

          <a target="_blank"
            :href="`https://www.bsigroup.com/en-GB/search-results/?q='${documentName}'&standards=Standards`"
            class="flex  flex-grow md:flex-grow-0 ">
            <button type="button"
              class="flex-grow px-6 py-2.5 bg-blue-600 text-white font-medium text-xs leading-tight uppercase rounded shadow-md hover:bg-blue-700 hover:shadow-lg focus:bg-blue-700 focus:shadow-lg focus:outline-none focus:ring-0 active:bg-blue-800 active:shadow-lg transition duration-150 ease-in-out">View
              this standard</button>
          </a>
        </div>

      </div>
      <p class="text-gray-500 text-base my-4 block break-all md:break-words ">
        <span class="inline-block align-bottom  text-xs ">
          Related wording:

          <i class="">
            {{ data[1].contents[0].filtered_NER_labels }}
          </i>
        </span>

      </p>


    </div>
  </div>

</template>

<style scoped>
</style>
