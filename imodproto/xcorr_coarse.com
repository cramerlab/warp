# THIS IS A COMMAND FILE TO RUN TILTXCORR AND DETERMINE CROSS-CORRELATION
# ALIGNMENT OF A TILT SERIES
#
#
# TO RUN TILTXCORR
#
####CreatedVersion####4.9.12
#
# Add BordersInXandY to use a centered region smaller than the default
# or XMinAndMax and YMinAndMax  to specify a non-centered region
#
$tiltxcorr -StandardInput
InputFile	$FILENAME$.st
OutputFile	$FILENAME$.prexf
TiltFile	$FILENAME$.rawtlt
RotationAngle	$AXISANGLE$
FilterSigma1	0.03
FilterRadius2	0.25
FilterSigma2	0.05
$if (-e ./savework) ./savework
