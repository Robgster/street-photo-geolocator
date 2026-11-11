# street-photo-geolocator

[![make.fun · photo-geolocator-01](https://make.fun/badge/photo-geolocator-01.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)

An agent for make.fun's photo-geolocator-01 bounty: for each street-level photo it asks Claude Opus 5.5 to read the clues in the picture (driving side, road markings, signs and script, bollards, poles, plates, vegetation, sun) and commit to a latitude and longitude. A budget ladder keeps the trial inside its $1.00: the biggest photos go first, calls go out in rounds that fit the gateway's worst-case reservation, and a photo steps down to Sonnet 5.5, then Haiku 5.5, only when Opus no longer fits or gives no answer. Every photo gets an answer.

Node 22, standard library only, deterministic given the model's answers (rounds depend only on earlier rounds' results).

```sh
MAKEFUN_ENV_URL=... ANTHROPIC_BASE_URL=... ANTHROPIC_API_KEY=... node main.ts
```
