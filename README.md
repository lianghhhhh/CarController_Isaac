# CarController_Isaac
use a nn model to control a car run on a path in Isaac Sim

## Environment
WSL Ubuntu 22.04.5 LTS + Windows Isaac Sim 4.5.0

## WSL ROS2
1. build packages
```
cd CarController_Isaac
colcon build
. ./install/setup.bash
```
2. run control car node
```
 ros2 run control_car_pkg control_car_node
```

## Isaac Sim
1. run ***isaac-sim.selector.bat***
2. pull ***pros_car.usd*** into Isaac Sim
3. at Stage on the right, click ***base_link*** & ***BasisCurves***
4. press ***Play***
