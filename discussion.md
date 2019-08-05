    本代码中提供的方法为模板匹配，小图为模板，以滑动窗口的形式在大图上进行匹配（代码中使用的是平方差来进行匹配，平方差越小两图的匹配程度越高，最后调节阈值来确定最终的匹配结果）。由于只使用了灰度信息，因而遮挡、光照、形变、旋转、尺度变化对模型的匹配结果影响很大，解决以上问题，可以使用对以上干扰具有一定抵抗作用的局部特征和描述子来进行匹配程度的计算，也可以使用多尺度模板来解决待检物体多尺度的问题。如果样本足够多，可以对描述子进行机器学习来进行匹配，进一步可以设计基于深度学习端到端的定位方法。