# Reference English FUTO model — user-defined symbol slot map

Source: `tokenizer.ggml.tokens`, indices 4..303 (300 slots, type=USER_DEFINED).
These are the structural / special tokens. The pt_BR model must preserve
the **structural** slot indices (174..207: XBU/XBC/XEC/XC0-4/CHAR_A-Z) and may
replace the **content** slots (contractions, common words, emoji) with pt_BR equivalents.
Whether the keyboard app indexes structural tokens by *name lookup* in
`tokenizer.ggml.tokens` or by *fixed integer ID* is unclear — preserving the same
indices is the safe choice and what this plan does.

## Indices 4..27: FUTO reserved/filler slots (likely unused — keep as <FUTO0>..<FUTO23>)

```
   4  '<FUTO0>'
   5  '<FUTO1>'
   6  '<FUTO2>'
   7  '<FUTO3>'
   8  '<FUTO4>'
   9  '<FUTO5>'
  10  '<FUTO6>'
  11  '<FUTO7>'
  12  '<FUTO8>'
  13  '<FUTO9>'
  14  '<FUTO10>'
  15  '<FUTO11>'
  16  '<FUTO12>'
  17  '<FUTO13>'
  18  '<FUTO14>'
  19  '<FUTO15>'
  20  '<FUTO16>'
  21  '<FUTO17>'
  22  '<FUTO18>'
  23  '<FUTO19>'
  24  '<FUTO20>'
  25  '<FUTO21>'
  26  '<FUTO22>'
  27  '<FUTO23>'
```

## Indices 28..173: English contractions / common words / bigrams — REPLACE with pt_BR equivalents

```
  28  "'d▁"
  29  "'ll▁"
  30  "'re▁"
  31  "'s▁"
  32  "'ve▁"
  33  "I'd▁"
  34  "I'll▁"
  35  "I'm▁"
  36  "I've▁"
  37  "ain't▁"
  38  "aren't▁"
  39  "can't▁"
  40  "could've▁"
  41  "couldn't▁"
  42  "didn't▁"
  43  "doesn't▁"
  44  "don't▁"
  45  "hadn't▁"
  46  "hasn't▁"
  47  "haven't▁"
  48  "he'd▁"
  49  "he'll▁"
  50  "he's▁"
  51  "here's▁"
  52  "how'd▁"
  53  "how're▁"
  54  "how's▁"
  55  "i'd▁"
  56  "i'll▁"
  57  "i'm▁"
  58  "i've▁"
  59  "isn't▁"
  60  "it'd▁"
  61  "it'll▁"
  62  "it's▁"
  63  "let's▁"
  64  "might've▁"
  65  "must've▁"
  66  "mustn't▁"
  67  "needn't▁"
  68  "o'clock▁"
  69  "she'd▁"
  70  "she'll▁"
  71  "she's▁"
  72  "should've▁"
  73  "shouldn't▁"
  74  "that'd▁"
  75  "that'll▁"
  76  "that's▁"
  77  "there'd▁"
  78  "there'll▁"
  79  "there're▁"
  80  "there's▁"
  81  "they'd▁"
  82  "they'll▁"
  83  "they're▁"
  84  "they've▁"
  85  "this'll▁"
  86  "wasn't▁"
  87  "we'd▁"
  88  "we'll▁"
  89  "we're▁"
  90  "we've▁"
  91  "weren't▁"
  92  "what'd▁"
  93  "what'll▁"
  94  "what're▁"
  95  "what's▁"
  96  "where'd▁"
  97  "where's▁"
  98  "who'd▁"
  99  "who'll▁"
 100  "who're▁"
 101  "who's▁"
 102  "who've▁"
 103  "why'd▁"
 104  "won't▁"
 105  "would've▁"
 106  "wouldn't▁"
 107  "you'd▁"
 108  "you'll▁"
 109  "you're▁"
 110  "you've▁"
 111  '.▁'
 112  ',▁'
 113  '?▁'
 114  '!▁'
 115  '...▁'
 116  '-▁'
 117  ')▁'
 118  ']▁'
 119  '>▁'
 120  '+▁'
 121  '"▁'
 122  ':▁'
 123  ';▁'
 124  '=▁'
 125  '%▁'
 126  '\t'
 127  '\n'
 128  '\x0b'
 129  '\x0c'
 130  '\r'
 131  ' '
 132  '!'
 133  '"'
 134  '#'
 135  '$'
 136  '%'
 137  '&'
 138  "'"
 139  '('
 140  ')'
 141  '*'
 142  '+'
 143  ','
 144  '-'
 145  '.'
 146  '/'
 147  '0'
 148  '1'
 149  '2'
 150  '3'
 151  '4'
 152  '5'
 153  '6'
 154  '7'
 155  '8'
 156  '9'
 157  ':'
 158  ';'
 159  '<'
 160  '='
 161  '>'
 162  '?'
 163  '@'
 164  '['
 165  '\\'
 166  ']'
 167  '^'
 168  '_'
 169  '`'
 170  '{'
 171  '|'
 172  '}'
 173  '~'
```

## Indices 174..176: STRUCTURAL: <XBU>/<XBC>/<XEC> — autocorrect format markers (KEEP IDENTICAL)

```
 174  '<XBU>'
 175  '<XBC>'
 176  '<XEC>'
```

## Indices 177..181: STRUCTURAL: <XC0>..<XC4> — swipe-typing markers (KEEP IDENTICAL)

```
 177  '<XC0>'
 178  '<XC1>'
 179  '<XC2>'
 180  '<XC3>'
 181  '<XC4>'
```

## Indices 182..207: STRUCTURAL: <CHAR_A>..<CHAR_Z> — per-key tokens (KEEP IDENTICAL)

```
 182  '<CHAR_A>'
 183  '<CHAR_B>'
 184  '<CHAR_C>'
 185  '<CHAR_D>'
 186  '<CHAR_E>'
 187  '<CHAR_F>'
 188  '<CHAR_G>'
 189  '<CHAR_H>'
 190  '<CHAR_I>'
 191  '<CHAR_J>'
 192  '<CHAR_K>'
 193  '<CHAR_L>'
 194  '<CHAR_M>'
 195  '<CHAR_N>'
 196  '<CHAR_O>'
 197  '<CHAR_P>'
 198  '<CHAR_Q>'
 199  '<CHAR_R>'
 200  '<CHAR_S>'
 201  '<CHAR_T>'
 202  '<CHAR_U>'
 203  '<CHAR_V>'
 204  '<CHAR_W>'
 205  '<CHAR_X>'
 206  '<CHAR_Y>'
 207  '<CHAR_Z>'
```

## Indices 208..263: More common English words/bigrams — REPLACE with pt_BR equivalents

```
 208  '_EV0▁'
 209  '_EV1▁'
 210  '_EV2▁'
 211  '_EV3▁'
 212  '_EV4▁'
 213  '_EV5▁'
 214  '_EV6▁'
 215  '_EV7▁'
 216  '_EV8▁'
 217  '_EV9▁'
 218  '_EV10▁'
 219  '_EV11▁'
 220  '_EV12▁'
 221  '_EV13▁'
 222  '_EV14▁'
 223  '_EV15▁'
 224  '_EV16▁'
 225  '_EV17▁'
 226  '_EV18▁'
 227  '_EV19▁'
 228  '_EV20▁'
 229  '_EV21▁'
 230  '_EV22▁'
 231  '_EV23▁'
 232  '_EV24▁'
 233  '_EV25▁'
 234  '_EV26▁'
 235  '_EV27▁'
 236  '_EV28▁'
 237  '_EV29▁'
 238  '_EV30▁'
 239  '_EV31▁'
 240  '😂'
 241  '❤'
 242  '😭'
 243  '😍'
 244  '😊'
 245  '😔'
 246  '💕'
 247  '😘'
 248  '😒'
 249  '😩'
 250  '😁'
 251  '🔥'
 252  '🙏'
 253  '☺'
 254  '👍'
 255  '😅'
 256  '👀'
 257  '😉'
 258  '👌'
 259  '😏'
 260  '✨'
 261  '💔'
 262  '😌'
 263  '😎'
```

## Indices 264..303: Emoji — can keep or curate for pt_BR audience (40 slots)

```
 264  '💜'
 265  '💙'
 266  '✅'
 267  '😢'
 268  '😳'
 269  '💯'
 270  '💖'
 271  '🎶'
 272  '🙌'
 273  '⬅'
 274  '😋'
 275  '🙈'
 276  '💀'
 277  '😄'
 278  '💗'
 279  '✌'
 280  '👉'
 281  '😞'
 282  '💛'
 283  '😜'
 284  '👏'
 285  '😑'
 286  '😆'
 287  '😴'
 288  '🌹'
 289  '😐'
 290  '😪'
 291  '😕'
 292  '💪'
 293  '😀'
 294  '💞'
 295  '😡'
 296  '💚'
 297  '🎉'
 298  '😱'
 299  '👇'
 300  '😈'
 301  '😃'
 302  '🌸'
 303  '💋'
```

