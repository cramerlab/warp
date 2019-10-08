# THIS IS A COMMAND FILE TO PRODUCE A PRE-ALIGNED STACK
# 
# The stack will be floated and converted to bytes under the assumption that
# you will go back to the raw stack to make the final aligned stack
#
$xftoxg
0
$FILENAME$.prexf
$FILENAME$.prexg
$newstack -StandardInput
InputFile	$FILENAME$.st
OutputFile	$FILENAME$.preali
TransformFile	$FILENAME$.prexg
FloatDensities	2
BinByFactor	$BINNINGFACTOR$
#DistortionField	.idf
ImagesAreBinned	1.0
AntialiasFilter	-1
#GradientFile	$FILENAME$.maggrad
$if (-e ./savework) ./savework
