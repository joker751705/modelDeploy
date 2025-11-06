import { createRouter, createWebHistory } from 'vue-router'
import ImagePredict from "@/components/ImagePredict.vue";

const routes = [
    {
        path: '/image',
        name: 'Image',
        component: ImagePredict
    }
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

export default router