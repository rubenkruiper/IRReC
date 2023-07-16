<template>
    <nav aria-label="Page navigation example" class="flex flex-col items-center mb-12 z-50">
        <ul class="inline-flex -space-x-px">
            <li>
                <button
                    class="flex-grow px-6 py-2.5 bg-blue-600 text-white font-medium text-xs leading-tight uppercase rounded-l shadow-md hover:bg-blue-700 hover:shadow-lg focus:shadow-lg focus:outline-none focus:ring-0 active:bg-blue-800 active:shadow-lg transition duration-150 ease-in-out"
                    type="button" @click="onClickFirstPage" :disabled="isInFirstPage">
                    First
                </button>
            </li>

            <li>
                <button
                    class="flex-grow px-6 py-2.5 bg-blue-600 text-white font-medium text-xs leading-tight uppercase  shadow-md hover:bg-blue-700 hover:shadow-lg  focus:shadow-lg focus:outline-none focus:ring-0 active:bg-blue-800 active:shadow-lg transition duration-150 ease-in-out"
                    type="button" @click="onClickPreviousPage" :disabled="isInFirstPage">
                    Previous
                </button>
            </li>

            <!-- Visible Buttons Start -->

            <li v-for="page in pages" :key="page.name">
                <button :class="{ 'bg-blue-800': currentPage === page.name }"
                    class="flex-grow px-6 py-2.5 bg-blue-600 text-white font-medium text-xs leading-tight uppercase shadow-md hover:bg-blue-700 hover:shadow-lg focus:shadow-lg focus:outline-none focus:ring-0 active:bg-blue-800 active:shadow-lg transition duration-150 ease-in-out"
                    type="button" :disabled="page.isDisabled" @click="onClickPage(page.name)">
                    {{ page.name }}
                </button>
            </li>

            <!-- Visible Buttons End -->

            <li>
                <button
                    class="flex-grow px-6 py-2.5 bg-blue-600 text-white font-medium text-xs leading-tight uppercase shadow-md hover:bg-blue-700 hover:shadow-lg  focus:shadow-lg focus:outline-none focus:ring-0 active:bg-blue-800 active:shadow-lg transition duration-150 ease-in-out"
                    type="button" @click="onClickNextPage" :disabled="isInLastPage">
                    Next
                </button>
            </li>

            <li>
                <button
                    class="flex-grow px-6 py-2.5 bg-blue-600 text-white font-medium text-xs leading-tight uppercase rounded-r shadow-md hover:bg-blue-700 hover:shadow-lg focus:shadow-lg focus:outline-none focus:ring-0 active:bg-blue-800 active:shadow-lg transition duration-150 ease-in-out"
                    type="button" @click="onClickLastPage" :disabled="isInLastPage">
                    Last
                </button>
            </li>
        </ul>
    </nav>
</template>

<script>

import { defineComponent, ref, computed, onMounted, watch } from 'vue'
import { SearchStore } from '@/store/SearchStore'

export default defineComponent({
    props: {
        maxVisibleButtons: {
            type: Number,
            required: false,
            default: 3
        },
        totalPages: {
            type: Number,
            required: true
        },
        perPage: {
            type: Number,
            required: true
        },

    },
    setup(props, { emit }) {

        const store = SearchStore();
        const currentPage = computed(() => { return store.getCurrentPage });
        const startPage = computed(() => {
            // When on the first page
            if (currentPage.value === 1) {
                return 1;
            }
            // When on the last page
            if (currentPage.value === props.totalPages && props.totalPages > 2) {
                return props.totalPages - props.maxVisibleButtons;
            }

            // When inbetween
            return currentPage.value - 1;
        });

        const pages = computed(() => {
            const range = [];

            for (
                let i = startPage.value;
                i <= Math.min(startPage.value + props.maxVisibleButtons - 1, props.totalPages);
                i++
            ) {
                range.push({
                    name: i,
                    isDisabled: i === currentPage.value
                });
            }

            return range;
        });
        const isInFirstPage = computed(() => { return currentPage.value === 1 });
        const isInLastPage = computed(() => { return currentPage.value === props.totalPages });


        function onClickFirstPage() {
            emit('pagechanged', 1);
        }
        function onClickPreviousPage() {
            emit('pagechanged', currentPage.value - 1);
        }
        function onClickPage(page) {
            emit('pagechanged', page);
        }
        function onClickNextPage() {
            emit('pagechanged', currentPage.value + 1);
        }
        function onClickLastPage() {
            emit('pagechanged', props.totalPages);
        }
        return {

            store,
            currentPage,
            onClickLastPage,
            onClickNextPage,
            onClickPage,
            onClickPreviousPage,
            onClickFirstPage,
            isInLastPage,
            isInFirstPage,
            pages,
            startPage,

        }
    }

})


</script>