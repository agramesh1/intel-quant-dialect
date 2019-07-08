// RUN: mlir-opt %s -quantizer-infer-quantized-types -quant-convert-const -quantizer-remove-instrumentation -canonicalize -quantizer-conv2d-fusion -split-input-file | FileCheck %s

// ----
// Conv2DRequantize.
// CHECK-LABEL: @conv2d_requantize
func @conv2d_requantize(%arg0: tensor<300x3xf32>) -> tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %1 = "fxpmath.real_conv2d"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  %2 = "quant.qcast"(%1) : (tensor<300x5xf32>) -> tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>
  return %2 : tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>
}

// ----
// Conv2DRelu.
// CHECK-LABEL: @conv2d_relu
func @conv2d_relu(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %1 = "fxpmath.real_conv2d"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  %2 = "fxpmath.real_relu"(%1){clamp_max:6} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  return %2 : tensor<300x5xf32>
}

// ----
// Conv2DReluRequantize.
// CHECK-LABEL: @conv2d_relu_requantize
func @conv2d_relu_requantize(%arg0: tensor<300x3xf32>) -> tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %1 = "fxpmath.real_conv2d"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  %2 = "fxpmath.real_relu"(%1){clamp_max:6} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  %3 = "quant.qcast"(%2) : (tensor<300x5xf32>) -> tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>
  return %3 : tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>
}

// ----
// Conv2DBiasRelu.
// CHECK-LABEL: @conv2d_bias_relu
func @conv2d_bias_relu(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %cst_0 = constant  {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %1 = "fxpmath.real_conv2d"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  %2 = "fxpmath.real_bias"(%1, %cst_0) : (tensor<300x5xf32>, tensor<5xf32>) -> tensor<300x5xf32>
  %3 = "fxpmath.real_relu"(%2){clamp_max:6} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  return %3 : tensor<300x5xf32>
}

// ----
// Conv2DBiasReluRequantize.
// CHECK-LABEL: @conv2d_bias_relu_requantize
func @conv2d_bias_relu_requantize(%arg0: tensor<300x3xf32>) ->  tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %cst_0 = constant  {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %1 = "fxpmath.real_conv2d"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  %2 = "fxpmath.real_bias"(%1, %cst_0) : (tensor<300x5xf32>, tensor<5xf32>) -> tensor<300x5xf32>
  %3 = "fxpmath.real_relu"(%2){clamp_max:6} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  %4 = "quant.qcast"(%3) : (tensor<300x5xf32>) -> tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>
  return %4 : tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>
}

// ----
// Conv2DBiasSumRelu.
// CHECK-LABEL: @conv2d_bias_sum_relu
func @conv2d_bias_sum_relu(%arg0: tensor<300x3xf32>, %arg1: tensor<300x5xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %1 = "quant.stats"(%arg1) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %cst_0 = constant  {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %2 = "fxpmath.real_conv2d"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  %3 = "fxpmath.real_bias"(%2, %cst_0) : (tensor<300x5xf32>, tensor<5xf32>) -> tensor<300x5xf32>
  %4 = "fxpmath.real_add"(%3, %1) : (tensor<300x5xf32>, tensor<300x5xf32>) -> tensor<300x5xf32>
  %5 = "fxpmath.real_relu"(%4){clamp_max:6} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  return %5 : tensor<300x5xf32>
}

// ----
// Conv2DBiasSumReluRequantize.
// CHECK-LABEL: @conv2d_bias_sum_relu_requantize
func @conv2d_bias_sum_relu_requantize(%arg0: tensor<300x3xf32>, %arg1: tensor<300x5xf32>) -> tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>> {
  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %1 = "quant.stats"(%arg1) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
  %cst_0 = constant  {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
  %2 = "fxpmath.real_conv2d"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  %3 = "fxpmath.real_bias"(%2, %cst_0) : (tensor<300x5xf32>, tensor<5xf32>) -> tensor<300x5xf32>
  %4 = "fxpmath.real_add"(%3, %1) : (tensor<300x5xf32>, tensor<300x5xf32>) -> tensor<300x5xf32>
  %5 = "fxpmath.real_relu"(%4){clamp_max:6} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  %6 = "quant.qcast"(%5) : (tensor<300x5xf32>) -> tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>
  return %6 : tensor<300x5x!quant.uniform<u8:f32, 0.037564418067230126:163>>
}

