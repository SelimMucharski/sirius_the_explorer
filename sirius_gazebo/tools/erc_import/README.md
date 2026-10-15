# ERC world import from unity_sim

Converts an ERC Marsyard scene from
[agh-space-systems-rover/unity_sim](https://github.com/agh-space-systems-rover/unity_sim)
into a Gazebo world (`worlds/erc<year>.sdf` + `models/`). The generated files are
committed, so this is only needed to regenerate them or import another year.

Requirements: `libassimp-dev`, `python3-numpy`, `python3-pil`.

```bash
cd sirius_gazebo/tools/erc_import
mkdir -p build && g++ -O2 -std=c++17 fbx2obj.cpp -o build/fbx2obj -lassimp
git clone --depth 1 https://github.com/agh-space-systems-rover/unity_sim.git /tmp/unity_sim
python3 unity_to_gazebo.py /tmp/unity_sim ../.. --year 2025
```

What it does:

* terrain: FBX -> OBJ (visual mesh with texture + low-poly collision mesh), converted
  from Unity's left-handed Y-up frame to ROS FLU (`x = Unity z`, `y = -Unity x`,
  `z = Unity y`), shifted down so the ground at the start point is at z ~= 0;
* ArUco tag posts with their dictionary/ID textures (upscaled with nearest-neighbour);
* the scene's calibration points (red spheres, visual only);
* `<spherical_coordinates>` fitted to the scene's GPS base stations, as unity_sim's
  `GPS.cs` does, so `/gps/fix` gives the competition coordinates.

The script aborts if the calibration points do not lie on the converted terrain.
Only `ERC2025` (separate collision mesh) has been tested; other years store the
terrain differently (`ERC2023` uses a Unity Terrain asset) and need extra work.
