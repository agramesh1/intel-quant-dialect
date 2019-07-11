include "tensorflow/compiler/mlir/tensorflow/transforms/optimize.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

def : Pat<(IntelQuant_RealConv2DOp
              (Quant_QuantizeCastOp
                  $input, $min, $max, $num_bits, $narrow_range),
              $perm),
          (IntelQuant_RealConv2DRequantizeOp 
                  $input, $min, $max, $num_bits, $narrow_range,
              $perm)>;
