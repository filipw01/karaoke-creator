interface RawUltrastarSectionProps {
  ultrastarTxt: string
}

export const RawUltrastarSection = (props: RawUltrastarSectionProps) => {
  return (
    <details class="bg-zinc-800/60 border border-zinc-700/50 rounded-xl">
      <summary class="text-white font-medium p-4 cursor-pointer hover:text-violet-300 transition-colors">
        Raw UltraStar TXT
      </summary>
      <pre class="p-4 pt-0 text-zinc-400 text-xs font-mono overflow-x-auto whitespace-pre">
        {props.ultrastarTxt}
      </pre>
    </details>
  )
}
