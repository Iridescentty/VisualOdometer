# ***实现了简单的视觉里程计***

**数据在dataset/sequences/00/image_0/里**

## ***编译：***

**在cpp文件所在目录打开终端**

输入

```g++ slam_kitti.cpp slam_kitti_imp.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv | sed 's/ -lippicv//g'` `pkg-config --cflags eigen3` `pkg-config --libs eigen3` -o  VisualOdo```

```./VisualOdo```

参考：
https://github.com/avisingh599/mono-vo.git


