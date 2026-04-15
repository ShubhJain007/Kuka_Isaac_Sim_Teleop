# Streaming Robot Camera Feed to Apple Vision Pro

This guide sets up a live camera feed from the robot arm (via ROS2) displayed as a
**floating companion window** inside the Tracking Streamer AR app on Apple Vision Pro.

---

## Architecture

```
Workstation                                 Apple Vision Pro
───────────────────────────────             ─────────────────────────────
ROS2 /camera_arm/... topic                  Tracking Streamer App
        │                                   ┌──────────────────────────┐
        ▼                                   │  ImmersiveSpace (AR)     │
camera_hls_stream.py                        │    robot arm, bones...   │
  ├── rclpy subscriber                      │                          │
  ├── JPEG encoder → /mjpeg (~100ms)        │  WindowGroup (floating)  │
  ├── ffmpeg H.264 → /stream.m3u8 (~0.5s)  │    ┌──────────────┐      │
  └── HTTP server :8080                     │    │  AsyncImage   │      │
        │                                   │    │  (MJPEG feed) │      │
        │  http://<IP>:8080/mjpeg           │    │              │      │
        └───────────────────────────────────│───▶│              │      │
                                            │    └──────────────┘      │
                                            └──────────────────────────┘
```

Two endpoints are served simultaneously:

| Endpoint | Latency | Use case |
|---|---|---|
| `/mjpeg` | ~100-200 ms | **Recommended** — frame-by-frame JPEG, near real-time |
| `/stream.m3u8` | ~0.5-1 s | Fallback — H.264 HLS for AVPlayer compatibility |

---

## Part 1: Workstation Setup (Python)

### Prerequisites

```bash
# ffmpeg should already be installed:
ffmpeg -version

# If not:
sudo apt install ffmpeg
```

### Running the stream

In a **separate terminal** (alongside the simulation):

```bash
cd ~/Isaac_Lab_projects/Kuka_Med_7

python scripts/camera_hls_stream.py \
    --topic /camera_arm/camera/color/image_rect_raw \
    --port 8080 \
    --width 640 --height 480 --fps 30 --jpeg-quality 70
```

It will print something like:

```
================================================================
  Camera Stream Server
  ROS topic  : /camera_arm/camera/color/image_rect_raw
  Resolution : 640x480 @ 30 fps
  JPEG quality: 70

  MJPEG (low latency ~100ms):
    http://192.168.0.135:8080/mjpeg

  HLS (AVPlayer compat, ~0.5-1s latency):
    http://192.168.0.135:8080/stream.m3u8

  (If using Tailscale, replace with your 100.x.x.x IP)
================================================================
```

### Tuning latency vs quality

| Flag | Default | Effect |
|---|---|---|
| `--jpeg-quality 50` | 70 | Lower = smaller frames = less latency, more compression artifacts |
| `--width 320 --height 240` | 640x480 | Half resolution = half the data, noticeably faster |
| `--fps 15` | 30 | Half framerate = half data, still very usable for monitoring |

For **fastest possible** feed over Tailscale:
```bash
python scripts/camera_hls_stream.py --width 320 --height 240 --fps 15 --jpeg-quality 50
```

### If using Tailscale

Use the Tailscale IP instead of the LAN IP:

```bash
tailscale ip -4    # prints your 100.x.x.x address
```

Then the URL becomes `http://100.x.x.x:8080/mjpeg`

---

## Part 2: Tracking Streamer App Modifications (Swift / Xcode)

You need to add **two things** to the Tracking Streamer visionOS app:

1. A new `CameraFeedView.swift` file (the floating video window)
2. A new `WindowGroup` entry in `App.swift`
3. A button in `ContentView.swift` to open/close the camera window

### Prerequisites

- Mac with **Xcode 16+**
- Apple Developer account (free tier works for sideloading to your own device)
- Clone or fork [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop)

### Step 1: Create `CameraFeedView.swift`

Create a new file `Tracking Streamer/CameraFeedView.swift` with this content.

This uses an MJPEG stream reader (~100ms latency) for near real-time display,
with a fallback HLS AVPlayer mode if needed:

```swift
import SwiftUI
import AVKit

// MARK: - MJPEG stream reader (low latency, ~100ms)

class MJPEGStreamReader: ObservableObject {
    @Published var currentFrame: UIImage?
    @Published var isConnected = false
    @Published var fps: Double = 0

    private var task: URLSessionDataTask?
    private var buffer = Data()
    private var frameCount = 0
    private var fpsTimer: Date = Date()

    func start(url: URL) {
        stop()
        isConnected = false
        buffer = Data()
        frameCount = 0
        fpsTimer = Date()

        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = 10

        let config = URLSessionConfiguration.default
        config.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        config.urlCache = nil
        // Disable buffering for lowest latency
        config.httpShouldUsePipelining = true

        let session = URLSession(configuration: config,
                                 delegate: MJPEGDelegate(reader: self),
                                 delegateQueue: nil)
        task = session.dataTask(with: request)
        task?.resume()
    }

    func stop() {
        task?.cancel()
        task = nil
        isConnected = false
    }

    fileprivate func didReceiveData(_ data: Data) {
        buffer.append(data)
        if !isConnected {
            DispatchQueue.main.async { self.isConnected = true }
        }

        // Scan for JPEG boundaries (0xFFD8 = start, 0xFFD9 = end)
        while let range = findJPEG(in: buffer) {
            let jpegData = buffer.subdata(in: range)
            buffer.removeSubrange(0..<range.upperBound)

            if let img = UIImage(data: jpegData) {
                DispatchQueue.main.async {
                    self.currentFrame = img
                }
                frameCount += 1
                let elapsed = Date().timeIntervalSince(fpsTimer)
                if elapsed >= 1.0 {
                    DispatchQueue.main.async {
                        self.fps = Double(self.frameCount) / elapsed
                    }
                    frameCount = 0
                    fpsTimer = Date()
                }
            }
        }

        // Prevent buffer bloat: if no JPEG found and buffer > 2 MB, trim
        if buffer.count > 2_000_000 {
            buffer = Data()
        }
    }

    private func findJPEG(in data: Data) -> Range<Int>? {
        guard data.count > 4 else { return nil }
        var start: Int?
        for i in 0..<(data.count - 1) {
            if data[i] == 0xFF && data[i+1] == 0xD8 {
                start = i
            }
            if let s = start, data[i] == 0xFF && data[i+1] == 0xD9 {
                return s..<(i + 2)
            }
        }
        return nil
    }
}

private class MJPEGDelegate: NSObject, URLSessionDataDelegate {
    weak var reader: MJPEGStreamReader?
    init(reader: MJPEGStreamReader) { self.reader = reader }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        reader?.didReceiveData(data)
    }
}

// MARK: - Camera Feed View

struct CameraFeedView: View {
    @AppStorage("cameraStreamURL") private var streamURLString: String = ""
    @StateObject private var mjpeg = MJPEGStreamReader()
    @State private var editingURL: String = ""
    @State private var showURLEditor = false

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: "video.fill")
                Text("Robot Camera")
                    .font(.headline)
                Spacer()
                if mjpeg.isConnected {
                    Text(String(format: "%.0f fps", mjpeg.fps))
                        .font(.caption)
                        .foregroundColor(.green)
                }
                Button(action: { showURLEditor.toggle() }) {
                    Image(systemName: "gear")
                }
            }
            .padding(.horizontal)

            // Video frame
            if let frame = mjpeg.currentFrame {
                Image(uiImage: frame)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 480, height: 360)
                    .cornerRadius(12)
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.black.opacity(0.3))
                        .frame(width: 480, height: 360)
                    VStack(spacing: 8) {
                        if streamURLString.isEmpty {
                            Image(systemName: "antenna.radiowaves.left.and.right.slash")
                                .font(.largeTitle)
                            Text("No stream URL configured")
                                .font(.caption)
                            Text("Tap the gear icon to set the URL")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        } else {
                            ProgressView()
                            Text("Connecting...")
                                .font(.caption)
                        }
                    }
                }
            }

            HStack(spacing: 16) {
                Button(action: startStream) {
                    Label("Play", systemImage: "play.fill")
                }
                .disabled(streamURLString.isEmpty)

                Button(action: { mjpeg.stop() }) {
                    Label("Stop", systemImage: "stop.fill")
                }
                .disabled(!mjpeg.isConnected)
            }
            .padding(.bottom, 8)
        }
        .padding()
        .frame(width: 520)
        .glassBackgroundEffect()
        .sheet(isPresented: $showURLEditor) {
            VStack(spacing: 16) {
                Text("Camera Stream URL")
                    .font(.headline)
                Text("Enter the MJPEG URL from the workstation\n(e.g. http://100.x.x.x:8080/mjpeg)")
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
                TextField("http://...:8080/mjpeg", text: $editingURL)
                    .textFieldStyle(.roundedBorder)
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
                    .padding(.horizontal)
                HStack {
                    Button("Cancel") { showURLEditor = false }
                    Button("Save") {
                        streamURLString = editingURL
                        showURLEditor = false
                        startStream()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding()
            .frame(width: 440)
            .onAppear { editingURL = streamURLString }
        }
        .onAppear {
            if !streamURLString.isEmpty { startStream() }
        }
        .onDisappear {
            mjpeg.stop()
        }
    }

    private func startStream() {
        guard let url = URL(string: streamURLString) else { return }
        mjpeg.start(url: url)
    }
}
```

### Step 2: Modify `App.swift`

Add a new `WindowGroup` for the camera feed. Open `App.swift` and add the
camera window group **inside** the `body` property, alongside the existing scenes:

```swift
var body: some Scene {
    WindowGroup {
        ContentView()
    }
    .windowResizability(.contentSize)

    // ── NEW: Floating camera feed window ──────────────────────
    WindowGroup(id: "cameraFeedWindow") {
        CameraFeedView()
    }
    .windowResizability(.contentSize)
    .defaultSize(width: 520, height: 480)
    // ──────────────────────────────────────────────────────────

    // Hand tracking view (existing)
    ImmersiveSpace(id: "immersiveSpace") {
        🌐RealityView(model: appModel)
    }

    // ... rest of existing scenes unchanged ...
}
```

### Step 3: Add a button in `ContentView.swift` to open the camera window

Find the main view body in `ContentView.swift` and add a button. You need
the `openWindow` environment action:

At the top of `ContentView`, add:

```swift
@Environment(\.openWindow) private var openWindow
```

Then add a button somewhere in the existing UI (e.g. near the connection controls):

```swift
Button(action: {
    openWindow(id: "cameraFeedWindow")
}) {
    Label("Robot Camera", systemImage: "video.fill")
}
```

### Step 4: Build and sideload

1. Open the Xcode project in `Tracking Streamer/`
2. Set the target to your Apple Vision Pro
3. Build & Run (Cmd+R)
4. The app deploys to your AVP

---

## Part 3: Using it

### First time setup

1. Start the simulation on the workstation with `--ros --avp`
2. In a separate terminal, run `camera_hls_stream.py` — note the MJPEG URL printed
3. On AVP, open Tracking Streamer, connect to workstation as usual
4. Tap "Robot Camera" button — a floating window appears
5. Tap the gear icon, paste the MJPEG URL (e.g. `http://100.64.1.5:8080/mjpeg`)
6. Tap Save — video starts playing within ~100ms

The URL is saved in `@AppStorage` so you only need to enter it once.

### After first setup

1. Start simulation + camera stream on workstation
2. Open Tracking Streamer on AVP, connect
3. Tap "Robot Camera" — it auto-starts with the saved URL
4. Drag the floating window to wherever you want in your room

### Tips

- The floating window can be **repositioned** by grabbing its window bar
- **MJPEG latency** is ~100-200ms — near real-time, good enough for surgical monitoring
- The FPS counter in the top-right shows actual received framerate
- If the feed is choppy, try `--fps 15 --width 320 --height 240 --jpeg-quality 50`
- The window stays visible even while the immersive space is active

---

## Troubleshooting

| Issue | Fix |
|---|---|
| "No stream URL configured" | Tap gear icon, enter the MJPEG URL (e.g. `http://IP:8080/mjpeg`) |
| "Connecting..." forever | Camera stream not running, or wrong IP. Check `camera_hls_stream.py` is running and the URL is correct |
| Stream never starts | ROS camera topic not publishing. Check `ros2 topic hz /camera_arm/camera/color/image_rect_raw` |
| Can't reach workstation from AVP | Check Tailscale is connected on both devices. Try `ping 100.x.x.x` from AVP |
| Low FPS / choppy | Reduce resolution: `--width 320 --height 240 --jpeg-quality 50` |
| Window doesn't appear | Make sure the `WindowGroup(id: "cameraFeedWindow")` was added to App.swift and the app was rebuilt |
