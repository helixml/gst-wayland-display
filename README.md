# gst-wayland-display

A micro Wayland compositor that can be used as a Gstreamer plugin. Based
on [smithay](https://github.com/Smithay/smithay)

## Install

see [cargo-c](https://github.com/lu-zero/cargo-c)

```bash
git clone https://github.com/games-on-whales/gst-wayland-display.git
cd gst-wayland-display
# Install cargo-c if you don't have it already
cargo install cargo-c
# Build and install the plugin, by default under 
cargo cinstall --prefix=/usr/local
```

## GStreamer plugin

By default it'll install the plugin in `/usr/local/lib/gstreamer-1.0/libgstwaylanddisplaysrc.so`.

You can check if the plugin is picked up by calling:

```bash
GST_PLUGIN_PATH=/usr/local/lib/gstreamer-1.0 gst-inspect-1.0 waylanddisplaysrc
```

Example pipeline:

```bash
GST_PLUGIN_PATH=/usr/local/lib/gstreamer-1.0 gst-launch-1.0 waylanddisplaysrc ! 'video/x-raw,width=1280,height=720,format=RGBx,framerate=60/1' !  autovideosink
```

If this starts you should have a wayland socket under `$XDG_RUNTIME_DIR`

```
ls $XDG_RUNTIME_DIR 
 wayland-1
 wayland-1.lock
```

You should then be able to start any wayland process and use that socket

```bash 
WAYLAND_DISPLAY=wayland-1 weston-simple-egl
```

## Zero copy pipeline support

This plugin supports outputting **DMA buffers** in order to achieve a proper **zero-copy pipeline**.  
It'll negotiate the proper caps with downstream elements using Gstreamer, you can read more about it [in the official docs](https://gstreamer.freedesktop.org/documentation/additional/design/dmabuf.html?gi-language=c).

Example pipelines:
- Nvidia
```bash
gst-launch-1.0 waylanddisplaysrc  ! 'video/x-raw(memory:DMABuf),width=1920,height=1080,framerate=60/1' ! glupload ! glcolorconvert ! 'video/x-raw(memory:GLMemory), format=NV12' ! nvh265enc ! nvh265dec ! autovideosink
```
- AMD/Intel
```bash
gst-launch-1.0 waylanddisplaysrc ! 'video/x-raw(memory:DMABuf),width=1920,height=1080,framerate=60/1' ! vapostproc ! 'video/x-raw(memory:VAMemory), format=NV12' ! vah265enc ! vah265dec ! autovideosink
```

## C Bindings

CmakeLists.txt

```cmake
pkg_check_modules(libgstwaylanddisplay REQUIRED IMPORTED_TARGET libgstwaylanddisplay)
target_link_libraries(<YOUR_PROJECT_HERE> PUBLIC PkgConfig::libgstwaylanddisplay)
```

Include in your code:

```c
#include <libgstwaylanddisplay/libgstwaylanddisplay.h>
```

Example usage:

```c++
auto w_state = display_init("/dev/dri/renderD128"); // Pass a render node
        
display_add_input_device(w_state, "/dev/input/event20"); // Mouse
display_add_input_device(w_state, "/dev/input/event21"); // Keyboard

// Setting video as 1920x1080@60
auto video_info = gst_caps_new_simple("video/x-raw",
                                  "width", G_TYPE_INT, 1920,
                                  "height", G_TYPE_INT, 1080,
                                  "framerate", GST_TYPE_FRACTION, 60, 1,
                                  "format", G_TYPE_STRING, "RGBx",
                                  NULL);
display_set_video_info(w_state, video_info);

// Get a list of the devices needed, ex: ["/dev/dri/renderD128", "/dev/dri/card0"]
auto n_devices = display_get_devices_len(w_state);
const char *devs[n_devices];
display_get_devices(w_state, devs, n_devices);

// Get a list of the env vars needed, notably the wayland socket
// ex: ["WAYLAND_DISPLAY=wayland-1"]
auto n_envs = display_get_envvars_len(w_state);
const char *envs[n_envs];
display_get_envvars(w_state, envs, n_envs);

// Example of polling for new video data
GstBuffer * v_buffer;
while(true){
  v_buffer = display_get_frame(w_state);
  // TODO: do something with the video data
}

display_finish(w_state); // Cleanup
```