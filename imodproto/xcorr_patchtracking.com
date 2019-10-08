$goto doxcorr
$doxcorr:
$tiltxcorr -StandardInput
BordersInXandY	72,51
IterateCorrelations	1
ImagesAreBinned	$BINNINGFACTOR$
InputFile	$FILENAME$.preali
OutputFile	$FILENAME$_pt.fid
PrealignmentTransformFile	$FILENAME$.prexg
TiltFile	$FILENAME$.rawtlt
RotationAngle	$AXISANGLE$
FilterSigma1	0.03
FilterRadius2	0.25
FilterSigma2	0.05
SizeOfPatchesXandY	300,300
OverlapOfPatchesXandY	0.8,0.8
$dochop:
$imodchopconts -StandardInput
InputModel $FILENAME$_pt.fid
OutputModel $FILENAME$.fid
MinimumOverlap	4
AssignSurfaces 1
$if (-e ./savework) ./savework
