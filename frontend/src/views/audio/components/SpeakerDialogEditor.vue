<template>
  <div class="space-y-3">
    <div
      v-for="(line, index) in modelValue"
      :key="line.id"
      class="rounded-2xl border border-gray-200 bg-gray-50/80 p-4 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900/40"
    >
      <div class="flex items-center justify-between gap-3">
        <div class="text-sm font-semibold text-gray-950 [.dark_&]:text-white">{{ t('audio.speakerDialog.roleTitle', { index: index + 1 }) }}</div>
        <button
          type="button"
          class="rounded-full px-3 py-1 text-xs font-medium text-gray-500 transition hover:bg-gray-100 hover:text-gray-900 disabled:cursor-not-allowed disabled:opacity-50 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-white cursor-pointer"
          :disabled="modelValue.length <= 1"
          @click="removeLine(index)"
        >
          {{ t('audio.speakerDialog.remove') }}
        </button>
      </div>

      <div class="mt-3 grid gap-3 sm:grid-cols-2">
        <label v-if="mode === 'custom_voice'" class="block">
          <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.speakerDialog.speaker') }}</span>
          <select
            :value="line.speaker"
            class="mt-1 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100"
            @change="updateLine(index, { speaker: ($event.target as HTMLSelectElement).value })"
          >
            <option v-for="speaker in qwenSpeakerOptions" :key="speaker.value" :value="speaker.value">
              {{ speaker.label }}
            </option>
          </select>
        </label>
        <label class="block">
          <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.speakerDialog.language') }}</span>
          <select
            :value="line.language || ''"
            class="mt-1 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100"
            @change="updateLine(index, { language: ($event.target as HTMLSelectElement).value })"
          >
            <option v-for="language in qwenLanguageOptions" :key="language.value" :value="language.value">
              {{ language.label }}
            </option>
          </select>
        </label>
      </div>

      <div class="mt-3">
        <label class="block">
          <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">
            {{ mode === 'voice_design' ? t('audio.speakerDialog.voiceDesignLabel') : t('audio.speakerDialog.voiceDescOptional') }}
          </span>
          <input
            :value="line.instruct"
            class="mt-1 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100"
            :placeholder="mode === 'voice_design' ? t('audio.speakerDialog.voiceDesignPlaceholder') : t('audio.speakerDialog.voiceDescPlaceholder')"
            @input="updateLine(index, { instruct: ($event.target as HTMLInputElement).value })"
          >
        </label>
      </div>

      <label class="mt-3 block">
        <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.speakerDialog.dialogText') }}</span>
        <textarea
          :value="line.text"
          class="mt-1 min-h-24 w-full resize-y rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm leading-6 outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100"
          :placeholder="t('audio.speakerDialog.dialogPlaceholder')"
          @input="updateLine(index, { text: ($event.target as HTMLTextAreaElement).value })"
        />
      </label>
    </div>

    <button
      type="button"
      class="w-full rounded-2xl border border-dashed border-gray-300 px-4 py-3 text-sm font-medium text-gray-600 transition hover:border-indigo-300 hover:bg-indigo-50/50 hover:text-indigo-700 [.dark_&]:border-gray-700 [.dark_&]:text-gray-300 [.dark_&]:hover:border-indigo-500/50 [.dark_&]:hover:bg-indigo-950/20 [.dark_&]:hover:text-indigo-200 cursor-pointer"
      @click="addLine"
    >
      {{ t('audio.speakerDialog.addRole') }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { listTtsSpeakers, type TtsSpeakerOption } from '../../../api/audio'

export interface SpeakerDialogLine {
  id: string
  speaker: string
  text: string
  language?: string
  instruct?: string
}

type DialogVoiceMode = 'custom_voice' | 'voice_design'
type SpeakerSelectOption = { value: string; label: string; nativeLanguage?: string }

const fallbackQwenSpeakerOptions: SpeakerSelectOption[] = [
  { value: 'Vivian', label: 'Vivian - 明亮、略带锐气的年轻女声。中文', nativeLanguage: '中文' },
  { value: 'Serena', label: 'Serena - 温暖柔和的年轻女声。中文', nativeLanguage: '中文' },
  { value: 'Uncle_Fu', label: 'Uncle_Fu - 音色低沉醇厚的成熟男声。中文', nativeLanguage: '中文' },
  { value: 'Dylan', label: 'Dylan - 清晰自然的北京青年男声。中文（北京方言）', nativeLanguage: '中文（北京方言）' },
  { value: 'Eric', label: 'Eric - 活泼、略带沙哑明亮感的成都男声。中文（四川方言）', nativeLanguage: '中文（四川方言）' },
  { value: 'Ryan', label: 'Ryan - 富有节奏感的动态男声。英语', nativeLanguage: '英语' },
  { value: 'Aiden', label: 'Aiden - 清晰中频的阳光美式男声。英语', nativeLanguage: '英语' },
  { value: 'Ono_Anna', label: 'Ono_Anna - 轻快灵活的俏皮日语女声。日语', nativeLanguage: '日语' },
  { value: 'Sohee', label: 'Sohee - 富含情感的温暖韩语女声。韩语', nativeLanguage: '韩语' },
]

const { t } = useI18n()

const qwenSpeakerOptions = ref<SpeakerSelectOption[]>([...fallbackQwenSpeakerOptions])
const defaultSpeaker = fallbackQwenSpeakerOptions[0]?.value ?? 'Vivian'

const qwenLanguageOptions = computed(() => [
  { value: '', label: t('audio.speakerDialog.autoDetect') },
  { value: 'Chinese', label: t('audio.langZh') },
  { value: 'English', label: t('audio.langEn') },
  { value: 'Japanese', label: t('audio.langJa') },
  { value: 'Korean', label: t('audio.langKo') },
  { value: 'German', label: '德语' },
  { value: 'French', label: '法语' },
  { value: 'Russian', label: '俄语' },
  { value: 'Portuguese', label: '葡萄牙语' },
  { value: 'Spanish', label: '西班牙语' },
  { value: 'Italian', label: '意大利语' },
])

const props = defineProps<{
  modelValue: SpeakerDialogLine[]
  mode: DialogVoiceMode
}>()

const emit = defineEmits<{
  'update:modelValue': [value: SpeakerDialogLine[]]
}>()

function makeLine(index: number): SpeakerDialogLine {
  const options = qwenSpeakerOptions.value.length > 0 ? qwenSpeakerOptions.value : fallbackQwenSpeakerOptions
  const speakerOption = options[index % options.length]

  return {
    id: `speaker-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    speaker: speakerOption?.value ?? defaultSpeaker,
    language: '',
    text: '',
    instruct: '',
  }
}

function updateLine(index: number, patch: Partial<SpeakerDialogLine>) {
  const next = props.modelValue.map((item, i) => (i === index ? { ...item, ...patch } : item))
  emit('update:modelValue', next)
}

function addLine() {
  emit('update:modelValue', [...props.modelValue, makeLine(props.modelValue.length)])
}

function removeLine(index: number) {
  if (props.modelValue.length <= 1) return
  emit('update:modelValue', props.modelValue.filter((_, i) => i !== index))
}

function toSpeakerSelectOptions(items: TtsSpeakerOption[] | undefined): SpeakerSelectOption[] {
  return (items || []).reduce<SpeakerSelectOption[]>((acc, item) => {
    const value = String(item.name || '').trim()
    if (!value) return acc
    const description = String(item.description || '').trim()
    const language = String(item.language || '').trim()
    const detail = [description, language].filter(Boolean).join('。')
    acc.push({
      value,
      label: detail ? `${item.label || value} - ${detail}` : String(item.label || value),
      nativeLanguage: language || undefined,
    })
    return acc
  }, [])
}

onMounted(async () => {
  try {
    const catalog = await listTtsSpeakers()
    const options = toSpeakerSelectOptions(catalog.families?.qwen?.speakers)
    if (options.length > 0) qwenSpeakerOptions.value = options
  } catch (error) {
    console.warn('[audio-dialog] failed to load shared qwen speakers, using fallback options', error)
  }
})
</script>
