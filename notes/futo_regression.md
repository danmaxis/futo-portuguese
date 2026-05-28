# FUTO suggestion regression — root cause

**Symptom**: After a recent FUTO Android-keyboard update (May 2026), the keyboard no longer suggests a word until the user has typed at least one character. Empty-prefix NWP suggestions disappeared.

**Root cause**: Single-line FUTO upstream commit `533b902` (May 20 2026, Aleksandras Kostarevas, "Ignore blank pastText in complex case for composing wrapper to resolve #2023").

File: `java/src/org/futo/inputmethod/latin/InputConnectionInternalComposingWrapper.kt`, around line 261.

```diff
- if(pastText != null) {
+ if(pastText != null && (pastText.isNotEmpty() || lengthToFetch == 0)) {
```

The composing-wrapper's `getTextBeforeCursor` path is now skipped when `pastText` is the empty string. The intent was to harden the wrapper against a `pastText == ""` edge case (issue #2023), but the side effect is that when the user has just committed a word and is at the start of a new one — `pastText` is empty and `lengthToFetch > 0` — the composing wrapper no longer feeds the LM with the trailing context. The empty-prefix `PredictNextWord` call in `native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp` (lines 1197-1205) therefore never runs in the normal new-word case.

**Confirmation**: this is the only commit in the 25-commit range `d25e64b..origin/master` that touches the composing/text-before-cursor path. `939df16` ("Fix inverted boolean expression") is in `InputLogic.java` and only affects ENTER-key dispatch in non-text fields — unrelated. The other 23 commits are emoji handling, clipboard search, layout engine, voice input, etc.

**Impact on our training**: **none**. The LM is unchanged. The empty-prefix path inside the LM still exists and still works when called. v8.gguf does not need to be retrained or repackaged on this account.

**Recommended action**: comment on FUTO issue #2023 (or open a new issue referencing 533b902) reporting that the guard regressed empty-prefix NWP suggestions. A minimal repro: install 0.1.28 (or whichever release contains 533b902), commit a word, observe that no suggestions appear until typing one character; revert the guard locally and the suggestions return.

**Workaround for the user**: install the previous FUTO release (pre-533b902, e.g. tag 0.1.27 / 0.1.27-rc1) until upstream fixes it. The grown typo log (393 unique pairs vs v8's 343 — adjacency 33.7%, accent 32.2%, prefix_completion **15.5%**, hybrid 11.8%) shows the user has still been getting useful prefixed-suggestion behavior in real typing; the regression specifically hurts the empty-prefix slot, which is also a smaller share of practical wins anyway.
