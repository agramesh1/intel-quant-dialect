// RUN: mlir-opt %s -quantizer-infer-quantized-types-intel -quant-convert-const -quantizer-remove-instrumentation -canonicalize -split-input-file | FileCheck %s

// ----
// A conv2D without fused clamp or bias.
// CHECK-LABEL: @conv2d
// CHECK: %cst = constant dense<tensor<3x5xi8>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>
// CHECK-NEXT: %2 = "intelquant.real_conv2d"(%0, %1) : (tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>, tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>) -> tensor<300x5xf32>
func @conv2d(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %1 = "intelquant.real_conv2d"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  return %1 : tensor<300x5xf32>
}

// ----
// A requantized conv2D without fused clamp or bias.
// CHECK-LABEL: @conv2d_requantize
// CHECK: %cst = constant dense<tensor<3x5xi8>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>
// CHECK-NEXT: %2 = "intelquant.real_conv2d_requantize"(%0, %1) : (tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>, tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>) -> tensor<300x5x!quant.uniform<i8:f32, 0.0629921259842528:-1>>
// CHECK-NEXT: %3 = "quant.dcast"(%2) : (tensor<300x5x!quant.uniform<i8:f32, 0.0629921259842528:-1>>) -> tensor<300x5xf32>
func @conv2d_requantize(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %1 = "intelquant.real_conv2d_requantize"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  %2 = "quant.stats"(%1) {layerStats: dense<tensor<2xf32>, [-8.000000e+00, 8.000000e+00]>} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  return %2 : tensor<300x5xf32>
}

// ----
// A conv2D with relu without fused clamp or bias.
// CHECK-LABEL: @conv2d_relu
// CHECK: %cst = constant dense<tensor<3x5xi8>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>
// CHECK-NEXT: %2 = "intelquant.real_conv2d_relu"(%0, %1) : (tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>, tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>) -> tensor<300x5xf32>
func @conv2d_relu(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %1 = "intelquant.real_conv2d_relu"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  return %1 : tensor<300x5xf32>
}

// ----
// A requantized conv2D with relu without fused clamp or bias.
// CHECK-LABEL: @conv2d_relu_requantize
// CHECK: %cst = constant dense<tensor<3x5xi8>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>
// CHECK-NEXT: %2 = "intelquant.real_conv2d_relu_requantize"(%0, %1) : (tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>, tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>) -> tensor<300x5x!quant.uniform<i8:f32, 0.0629921259842528:-1>>
// CHECK-NEXT: %3 = "quant.dcast"(%2) : (tensor<300x5x!quant.uniform<i8:f32, 0.0629921259842528:-1>>) -> tensor<300x5xf32>
func @conv2d_relu_requantize(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %1 = "intelquant.real_conv2d_relu_requantize"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  %2 = "quant.stats"(%1) {layerStats: dense<tensor<2xf32>, [-8.000000e+00, 8.000000e+00]>} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  return %2 : tensor<300x5xf32>
}

// ----
// A conv2D with fused bias.
// CHECK-LABEL: @conv2d_bias
// CHECK: %cst = constant dense<tensor<3x5xi8>
// CHECK-NEXT: %cst_0 = constant {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>
// CHECK-NEXT: %2 = "intelquant.real_conv2d_bias"(%0, %1, %cst_0) : (tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>, tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>, tensor<5xf32>) -> tensor<300x5xf32>
func @conv2d_bias(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %cst_0 = constant  {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %1 = "intelquant.real_conv2d_bias"(%0, %cst, %cst_0) : (tensor<300x3xf32>, tensor<3x5xf32>, tensor<5xf32>) -> tensor<300x5xf32>
  return %1 : tensor<300x5xf32>
}

// ----
// A requantized conv2D with fused bias.
// CHECK-LABEL: @conv2d_bias_requantize
// CHECK: %cst = constant dense<tensor<3x5xi8>
// CHECK-NEXT: %cst_0 = constant dense<tensor<5xi32>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>
// CHECK-NEXT: %2 = "quant.scast"(%cst_0) : (tensor<5xi32>) -> tensor<5x!quant.uniform<i32:f32, 0.0629921259842528>>
// CHECK-NEXT: %3 = "intelquant.real_conv2d_bias_requantize"(%0, %1, %2) : (tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>, tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>, tensor<5x!quant.uniform<i32:f32, 0.0629921259842528>>) -> tensor<300x5x!quant.uniform<i8:f32, 0.0629921259842528:-1>>
// CHECK-NEXT: %4 = "quant.dcast"(%3) : (tensor<300x5x!quant.uniform<i8:f32, 0.0629921259842528:-1>>) -> tensor<300x5xf32>
func @conv2d_bias_requantize(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %cst_0 = constant  {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %1 = "intelquant.real_conv2d_bias_requantize"(%0, %cst, %cst_0) : (tensor<300x3xf32>, tensor<3x5xf32>, tensor<5xf32>) -> tensor<300x5xf32>
  %2 = "quant.stats"(%1) {layerStats: dense<tensor<2xf32>, [-8.000000e+00, 8.000000e+00]>} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  return %2 : tensor<300x5xf32>
}

// ----
// A conv2D with fused bias and relu.
// CHECK-LABEL: @conv2d_bias_relu
// CHECK: %cst = constant dense<tensor<3x5xi8>
// CHECK-NEXT: %cst_0 = constant {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>
// CHECK-NEXT: %2 = "intelquant.real_conv2d_bias_relu"(%0, %1, %cst_0) : (tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>, tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>, tensor<5xf32>) -> tensor<300x5xf32>
func @conv2d_bias_relu(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %cst_0 = constant  {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %1 = "intelquant.real_conv2d_bias_relu"(%0, %cst, %cst_0) : (tensor<300x3xf32>, tensor<3x5xf32>, tensor<5xf32>) -> tensor<300x5xf32>
  return %1 : tensor<300x5xf32>
}

// ----
// A requantized conv2D with fused bias and relu.
// CHECK-LABEL: @conv2d_bias_relu_requantize
// CHECK: %cst = constant dense<tensor<3x5xi8>
// CHECK-NEXT: %cst_0 = constant dense<tensor<5xi32>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>
// CHECK-NEXT: %2 = "quant.scast"(%cst_0) : (tensor<5xi32>) -> tensor<5x!quant.uniform<i32:f32, 0.0629921259842528>>
// CHECK-NEXT: %3 = "intelquant.real_conv2d_bias_relu_requantize"(%0, %1, %2) : (tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>, tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>, tensor<5x!quant.uniform<i32:f32, 0.0629921259842528>>) -> tensor<300x5x!quant.uniform<u8:f32, 0.0629921259842528:127>>
// CHECK-NEXT: %4 = "quant.dcast"(%3) : (tensor<300x5x!quant.uniform<u8:f32, 0.0629921259842528:127>>) -> tensor<300x5xf32>
func @conv2d_bias_relu_requantize(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %cst_0 = constant  {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %1 = "intelquant.real_conv2d_bias_relu_requantize"(%0, %cst, %cst_0) : (tensor<300x3xf32>, tensor<3x5xf32>, tensor<5xf32>) -> tensor<300x5xf32>
  %2 = "quant.stats"(%1) {layerStats: dense<tensor<2xf32>, [-8.000000e+00, 8.000000e+00]>} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  return %2 : tensor<300x5xf32>
}

// ----
// A conv2D with fused bias, sum and relu.
// CHECK-LABEL: @conv2d_bias_sum_relu
// CHECK: %cst = constant dense<tensor<3x5xi8>
// CHECK-NEXT: %cst_0 = constant {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %1 = "quant.qcast"(%arg1) : (tensor<300x5xf32>) -> tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %2 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>
// CHECK-NEXT: %3 = "intelquant.real_conv2d_bias_sum_relu"(%0, %2, %cst_0, %1) : (tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>, tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>, tensor<5xf32>, tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>) -> tensor<300x5xf32>
func @conv2d_bias_sum_relu(%arg0: tensor<300x3xf32>, %arg1: tensor<300x5xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %1 = "quant.stats"(%arg1) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %cst_0 = constant  {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %cst_1 = constant  {name: "constant.38"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %2 = "intelquant.real_conv2d_bias_sum_relu"(%0, %cst, %cst_0, %1) : (tensor<300x3xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<300x5xf32>) -> tensor<300x5xf32>
  return %2 : tensor<300x5xf32>
}

// ----
// A requantized conv2D with fused bias, sum and relu.
// CHECK-LABEL: @conv2d_bias_sum_relu_requantize
// CHECK: %cst = constant dense<tensor<3x5xi8>
// CHECK-NEXT: %cst_0 = constant dense<tensor<5xi32>, [16, 32, 48, 63, 79]>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %1 = "quant.qcast"(%arg1) : (tensor<300x5xf32>) -> tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>
// CHECK-NEXT: %2 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>
// CHECK-NEXT: %3 = "quant.scast"(%cst_0) : (tensor<5xi32>) -> tensor<5x!quant.uniform<i32:f32, 0.0629921259842528>>
// CHECK-NEXT: %4 = "intelquant.real_conv2d_bias_sum_relu_requantize"(%0, %2, %3, %1) : (tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>, tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>, tensor<5x!quant.uniform<i32:f32, 0.0629921259842528>>, tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>) -> tensor<300x5x!quant.uniform<u8:f32, 0.0629921259842528:127>>
// CHECK-NEXT: %5 = "quant.dcast"(%4) : (tensor<300x5x!quant.uniform<u8:f32, 0.0629921259842528:127>>) -> tensor<300x5xf32>
func @conv2d_bias_sum_relu_requantize(%arg0: tensor<300x3xf32>, %arg1: tensor<300x5xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %1 = "quant.stats"(%arg1) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %cst_0 = constant  {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %cst_1 = constant  {name: "constant.38"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %2 = "intelquant.real_conv2d_bias_sum_relu_requantize"(%0, %cst, %cst_0, %1) : (tensor<300x3xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<300x5xf32>) -> tensor<300x5xf32>
  %3 = "quant.stats"(%2) {layerStats: dense<tensor<2xf32>, [-8.000000e+00, 8.000000e+00]>} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  return %2 : tensor<300x5xf32>
}

