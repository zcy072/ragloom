import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import { removeLegacyAuthSession } from './authSession.js'

removeLegacyAuthSession()

const app = createApp(App)
app.mount('#app')
