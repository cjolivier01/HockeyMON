# Shadow and color grading

Tracking and stitching previews provide **Shadow Lift Percent** and **Shadow Lift Black Point** controls for the stitched output and each source camera. Save and reset use the same per-game configuration as the other color controls.

```yaml
rink:
  camera:
    color:
      shadow_lift: 50
      shadow_lift_black_point: false
```

Use `stitching.left.color` or `stitching.right.color` for a source camera. Missing or null settings disable shadow lift. A percentage from 0 to 100 raises Rec. 709 luma using the reference gamma curve from hstream; RGB ratios remain unchanged until a channel clips. The optional black-point setting adds a neutral toe below 60% luma. Exposure remains measured in EV: +1 doubles brightness.

`HmImageColorAdjust` preserves alpha and leaves transparent pixels untouched. It supports uint16 images and explicit signal ranges through `input_max_value` (for example, 1 for normalized floats or 1023 for unpacked 10-bit samples). These transform capabilities do not change the current uint8 video decoder/encoder formats. Direct transform callers use RGB by default; the video pipeline sets BGR to match its decoded frames.
