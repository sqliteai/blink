# BlinkPilot

A driving demo that runs on Blink's WebAssembly build. The whole thing is
static files: no server, no API key, no network call per decision.

**The trip:** start in Millbrook, stop at the sign and the light, take the
on-ramp, merge onto Interstate 08, move over for Exit 14 and park in Cedar
Town.

The quickest way to try it needs no server and no Python:

```bash
make pilot                  # build/blinkpilot.html, about 770 KB
open build/blinkpilot.html  # or double-click it
```

`make pilot` inlines the page, its modules, `blink.wasm` and the model into one
HTML file. It opens straight from disk and can be put on any static host
(GitHub Pages, a bucket, an attachment) as it is. Browsers refuse `fetch` and
module imports over `file://`, which is the only reason the unbundled page
below needs a local HTTP server.

To retrain the model and work on the unbundled page:

```bash
make wasm
node examples/pilot/make_data.mjs data/pilot
PYTHONPATH=python .venv/bin/python -m blink_train.train data/pilot/train.jsonl \
    --validation data/pilot/validation.jsonl --preset tiny --epochs 8 --seed 7 \
    --output artifacts/blink-tiny-pilot-s7.blink
python3 -m http.server 8000
open http://localhost:8000/examples/pilot/
```

`?model=path.blink` loads another container, `?start=2500` starts the first
trip that many metres along the route (2500 is the interstate). If the model
cannot be loaded, the page drives with the oracle and says so.

**J** toggles the autopilot · **W S** throttle and brake · **A D** change lane
· **Space** brake · **C** candidate paths · **P** pause · **R** restart.
Driving by hand turns the autopilot off; the model keeps scoring, so you can
compare its pick with yours.

## How it works

| file | what it does |
|---|---|
| `world.mjs` | route, lanes, signals, traffic (IDM), the planner, the state text, the oracle |
| `make_data.mjs` | drives trips with the oracle plus random maneuvers and writes the JSONL corpus |
| `drive.mjs` | headless trips through the wasm runtime: blink, oracle, random-safe, random |
| `render.mjs` | top-down canvas renderer, HUD and minimap |
| `index.html` | the page: decision loop, probabilities, the text sent to the model |
| `bundle.mjs` | inlines everything into one HTML file (`make pilot`) |

Every 0.25 s the planner rolls each of six maneuvers three seconds ahead:

- `faster` accelerates toward the limit, `keep` holds speed, `slower` brakes
  gently.
- `stop` stops at the next required line, the end of the lane or 3 m behind
  the car ahead, and creeps up if it stopped short.
- `left` and `right` change lane at the current speed.

Each rollout runs against predicted traffic, which reacts to the ego car, and
against the signals' timing. Then the end state is checked: can the car still
stop before a red light, a stop sign, the end of its lane or the destination?
Is the car ahead too close to avoid? Would a lane change force the follower to
brake hard? Is the car too fast for a lower limit ahead? The outcome is one
sentence per maneuver, in shuffled order and with varied wording:

```
slower is safe but a bit slower. right drives off the road. stop is fine but
slower. keep makes the most progress. faster hits a car. left drifts off the
route lane.
```

The oracle ranks outcomes (most progress, then safe but slower, then off the
route lane, following too closely, too fast, illegal, cut-off, crash, no lane)
and labels the corpus. The model sees only the text above, the question and the
six maneuver names.

## Limits

BlinkPilot is a 2D scene with a fixed menu of six maneuvers, and the planner
turns each one's measurements into a sentence. blink-tiny reads at most 256
bytes of state, which is enough for six short sentences and not for richer
tables of candidates. There is no off-road recovery and no rerouting.
