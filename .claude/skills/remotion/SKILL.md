---
name: remotion
description: >
  Expert Remotion (React-based programmatic video generation) skill. Use whenever the
  user wants to create, edit, or build videos with code using Remotion; generate
  animated video content programmatically; build motion graphics, explainer videos,
  data visualizations as video, or social media clips with code. Trigger on: "Remotion",
  "programmatic video", "React video", "animated video with code", "video generation",
  "motion graphics code", "renderMedia", "useCurrentFrame", "Sequence", "interpolate",
  or any request to build a video using React/JavaScript. Also trigger when user has
  a Remotion project and needs help with animations, timing, composition, or rendering.

  Use this skill only when relevant to the task. Stay focused and minimal.
---

# Remotion Skill

You are an expert in Remotion — the framework for creating videos programmatically with React. You build stunning, high-quality animated videos using React components, Remotion's animation primitives, and compositing tools. Every video you produce is smooth, well-timed, and visually polished.

---

## Remotion Fundamentals

Remotion is React + a timeline. Every frame is a React render. There is no runtime playback — each frame is rendered deterministically, meaning the same frame number always produces the same pixel output.

### Key Concepts
```
fps               — Frames per second (30 for standard, 60 for smooth motion)
durationInFrames  — Total length in frames (30fps × 5s = 150 frames)
frame             — Current frame number, from useCurrentFrame()
interpolate       — Maps an input range (frames) to an output range (values)
spring            — Physics-based spring animation (natural-feeling bounce)
Sequence          — Mounts/unmounts a component within a frame range
Series            — Sequences that play one after another automatically
AbsoluteFill      — A div that fills 100% width/height, positioned absolutely
```

### Project Structure
```
my-video/
├── src/
│   ├── Root.tsx            ← Register all compositions here
│   ├── Composition.tsx     ← Main video composition
│   ├── scenes/
│   │   ├── Intro.tsx       ← Opening scene
│   │   ├── MainContent.tsx ← Core content
│   │   └── Outro.tsx       ← Closing scene + CTA
│   ├── components/
│   │   ├── AnimatedTitle.tsx
│   │   ├── CountUp.tsx
│   │   └── ProgressBar.tsx
│   └── lib/
│       └── animations.ts   ← Shared animation helpers (clamp config, etc.)
├── public/                 ← Static assets: fonts, images, audio
└── remotion.config.ts
```

---

## Component Structure

### Root Registration
```tsx
// src/Root.tsx
import { Composition } from 'remotion';
import { MyVideo } from './Composition';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="MyVideo"
        component={MyVideo}
        durationInFrames={300}   // 10 seconds at 30fps
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{
          title: 'Hello World',
          accent: '#4f8ef7',
        }}
      />
    </>
  );
};
```

### Basic Composition
```tsx
// src/Composition.tsx
import { AbsoluteFill, Sequence, useCurrentFrame, useVideoConfig } from 'remotion';
import { Intro } from './scenes/Intro';
import { MainContent } from './scenes/MainContent';
import { Outro } from './scenes/Outro';

interface Props { title: string; accent: string; }

export const MyVideo: React.FC<Props> = ({ title, accent }) => {
  const { fps } = useVideoConfig();
  const sec = (s: number) => Math.round(s * fps);

  return (
    <AbsoluteFill style={{ backgroundColor: '#000a64' }}>
      {/* Intro: 0–2s */}
      <Sequence from={0} durationInFrames={sec(2)}>
        <Intro title={title} accent={accent} />
      </Sequence>

      {/* Main content: 1.5s–8s (overlaps intro by 0.5s for smooth transition) */}
      <Sequence from={sec(1.5)} durationInFrames={sec(6.5)}>
        <MainContent accent={accent} />
      </Sequence>

      {/* Outro: 8s–10s */}
      <Sequence from={sec(8)} durationInFrames={sec(2)}>
        <Outro />
      </Sequence>
    </AbsoluteFill>
  );
};
```

---

## Animation Techniques

### `interpolate` — The Core Primitive
Maps an input range (frame numbers) to an output range (any values):

```tsx
import { interpolate, Easing, useCurrentFrame } from 'remotion';

const frame = useCurrentFrame();

// Shared clamp config (use everywhere to prevent runaway values)
const clamp = { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' } as const;

// Fade in over 20 frames
const opacity = interpolate(frame, [0, 20], [0, 1], clamp);

// Slide up: starts 40px below, rises to 0
const translateY = interpolate(frame, [0, 25], [40, 0], clamp);

// Multi-keyframe scale (appear with a bounce feel)
const scale = interpolate(frame, [0, 15, 20, 25], [0.8, 1.05, 0.97, 1.0], clamp);

// Eased progress (0→1 with ease-out)
const progress = interpolate(frame, [0, 60], [0, 1], {
  ...clamp,
  easing: Easing.out(Easing.quad),
});
```

**Always use `extrapolateLeft: 'clamp'` and `extrapolateRight: 'clamp'`** to prevent values running wild outside the defined keyframe range.

### `spring` — Physics-Based Animation
Natural-feeling motion without manually keyframing easing:

```tsx
import { spring, useCurrentFrame, useVideoConfig } from 'remotion';

const frame = useCurrentFrame();
const { fps } = useVideoConfig();

// Snappy pop-in
const scale = spring({
  frame,
  fps,
  from: 0,
  to: 1,
  config: {
    damping: 12,     // Higher = less bounce, settles faster
    stiffness: 200,  // Higher = faster to reach target
    mass: 0.5,       // Higher = heavier, slower
  },
});

// Delayed spring (start animation at frame 20)
const slideIn = spring({
  frame: frame - 20,  // Offset the start frame
  fps,
  from: -60,
  to: 0,
  config: { damping: 14, stiffness: 120 },
});
```

### Reusable Animation Components

```tsx
// FadeSlide — fade + slide up entrance with optional delay
const FadeSlide: React.FC<{
  children: React.ReactNode;
  delay?: number;
}> = ({ children, delay = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const f = frame - delay;

  const opacity = interpolate(f, [0, 18], [0, 1], { ...clamp });
  const y = spring({ frame: f, fps, from: 36, to: 0, config: { damping: 14, stiffness: 120 } });

  return (
    <div style={{ opacity, transform: `translateY(${y}px)` }}>
      {children}
    </div>
  );
};

// Typewriter — reveals text one character at a time
const Typewriter: React.FC<{ text: string; charsPerFrame?: number }> = ({
  text,
  charsPerFrame = 0.6,
}) => {
  const frame = useCurrentFrame();
  const visible = Math.min(text.length, Math.floor(frame * charsPerFrame));
  return <span>{text.slice(0, visible)}<span style={{ opacity: 0 }}>{text.slice(visible)}</span></span>;
};

// CountUp — animated number counter
const CountUp: React.FC<{ from: number; to: number; durationFrames: number }> = ({
  from, to, durationFrames,
}) => {
  const frame = useCurrentFrame();
  const progress = interpolate(frame, [0, durationFrames], [0, 1], {
    ...clamp,
    easing: Easing.out(Easing.cubic),
  });
  const value = Math.round(from + (to - from) * progress);
  return <span>{value.toLocaleString()}</span>;
};
```

---

## Timeline Control

### Sequence vs Series
```tsx
import { Sequence, Series } from 'remotion';

// Sequence: full manual control over from + duration
<Sequence from={60} durationInFrames={90}>
  <Scene2 />
</Sequence>

// Series: automatic back-to-back playback (cleaner for linear videos)
<Series>
  <Series.Sequence durationInFrames={60}><Intro /></Series.Sequence>
  <Series.Sequence durationInFrames={150}><MainSection /></Series.Sequence>
  <Series.Sequence durationInFrames={60}><Outro /></Series.Sequence>
</Series>
```

### Timing Helper (recommended in every project)
```tsx
// Put in lib/timing.ts
export const createTimer = (fps: number) => ({
  sec: (seconds: number) => Math.round(seconds * fps),
  ms: (milliseconds: number) => Math.round((milliseconds / 1000) * fps),
});

// Usage in composition
const { fps } = useVideoConfig();
const t = createTimer(fps);

<Sequence from={t.sec(2.5)} durationInFrames={t.sec(3)}>
  <Scene />
</Sequence>
```

### Audio & Background Music
```tsx
import { Audio, staticFile } from 'remotion';

// Background music (start from beginning, 40% volume)
<Audio src={staticFile('bg-music.mp3')} volume={0.4} />

// Voice-over starting at 1 second
<Audio src={staticFile('voiceover.mp3')} startFrom={0} delay={30} />

// Audio that fades out over last 30 frames
<Audio
  src={staticFile('bg-music.mp3')}
  volume={(f) => interpolate(f, [totalFrames - 30, totalFrames], [0.4, 0], clamp)}
/>
```

---

## Performance Tips

1. **Keep components pure** — same `frame` input → same output, always. No `Math.random()`, no `Date.now()`
2. **Use `delayRender` / `continueRender`** for any async loading (fonts, API, images):
   ```tsx
   const handle = delayRender('Loading font');
   loadFont().then(() => continueRender(handle));
   ```
3. **Use `<Img>` and `<Video>` from Remotion** — not native `<img>`/`<video>` — they handle asset preloading correctly
4. **Use `staticFile()`** for assets in the `public/` folder
5. **Memoize expensive calculations** with `useMemo` — frame-level calculations run 30× per second
6. **Concurrency for faster rendering**: `npx remotion render --concurrency 4`
7. **Preview at half resolution**: preview at 960×540, render at 1920×1080
8. **Frame range testing**: `--frames=0-60` to test just the intro before full render

---

## Video Storytelling Principles

Great programmatic videos follow clear narrative structures:

### The 3-Act Structure
```
Act 1 (15–20% of runtime): HOOK
  → State the problem or bold claim in the first 3 seconds
  → If the viewer isn't engaged in 3s, they're gone

Act 2 (60–70% of runtime): VALUE
  → Demonstrate the transformation / show the product / explain the concept
  → Use data, motion, and visuals to reinforce the message

Act 3 (15–20% of runtime): CTA
  → One clear action (subscribe, visit URL, try the product)
  → Repeat the core message as a single memorable line
```

### Scene Transition Patterns
```tsx
const clamp = { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' } as const;

// Cross-fade (both scenes visible during the overlap)
// Scene 1: opacity = 1 - progress
// Scene 2: opacity = progress
const progress = interpolate(frame, [50, 70], [0, 1], clamp);

// Wipe transition (clip-path slide)
const { width } = useVideoConfig();
const wipeX = interpolate(frame, [50, 80], [0, width], clamp);
// Scene 1: clipPath `inset(0 ${wipeX}px 0 0)`
// Scene 2: clipPath `inset(0 0 0 ${width - wipeX}px)`

// Scale punch-in (zoom into old scene as new one arrives)
const scaleOut = interpolate(frame, [50, 70], [1, 1.5], clamp);
const opacityOut = interpolate(frame, [60, 70], [1, 0], clamp);
```

### Motion Design Rules
- Enter from the natural reading direction (left → right, top → bottom)
- Exit in the opposite direction to entrance for continuity
- Hold important text/data for at least 1–2 seconds before animating it away
- Text should appear first, then data animates — never both at the same time
- Use consistent easing throughout a video — inconsistency feels unpolished
- Stagger related elements by 80–120ms for a natural cascade effect

---

## Rendering & Export

```bash
# Start the preview server
npx remotion preview

# Render to MP4
npx remotion render src/index.ts MyVideo out/video.mp4

# Render specific frame range (great for testing)
npx remotion render src/index.ts MyVideo out/clip.mp4 --frames=0-90

# Render a still frame (for thumbnails)
npx remotion still src/index.ts MyVideo --frame=30 out/thumbnail.png

# Render with custom props
npx remotion render src/index.ts MyVideo out/video.mp4 \
  --props='{"title":"Custom Title","accent":"#7c3aed"}'

# Faster render with concurrency
npx remotion render src/index.ts MyVideo out/video.mp4 --concurrency=4
```

### Output Formats
```
MP4 (H.264)   --codec h264    Best compatibility; default choice
WebM (VP9)    --codec vp9     Smaller file size; web-native
ProRes        --codec prores  Post-production editing workflows
GIF           --codec gif     Short loops < 5s; use sparingly (large files)
```
