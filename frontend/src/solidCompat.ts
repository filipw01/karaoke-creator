import { createEffect as coreCreateEffect } from "solid-js"

// Solid 2's native createEffect is split into compute/effect phases. This
// compatibility helper preserves the one-function effect style used throughout
// the migrated editor while keeping dependency tracking intact.
export const createEffect = (fn: () => void, _dependencies?: unknown) => {
  void _dependencies
  coreCreateEffect(
    () => {
      fn()
      return undefined
    },
    () => {},
  )
}
