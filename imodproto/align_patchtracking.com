# THIS IS A COMMAND FILE TO RUN TILTALIGN
#
####CreatedVersion####4.9.12
#
# To exclude views, add a line "ExcludeList view_list" with the list of views
#
# To specify sets of views to be grouped separately in automapping, add a line
# "SeparateGroup view_list" with the list of views, one line per group
#
$tiltalign -StandardInput
ModelFile	$FILENAME$.fid
ImageFile	$FILENAME$.preali
#ImageSizeXandY	$IMAGEWIDTH$,$IMAGEHEIGHT$
ImagesAreBinned	$BINNINGFACTOR$
OutputModelFile	$FILENAME$.3dmod
OutputResidualFile	$FILENAME$.resid
OutputFidXYZFile	$FILENAME$fid.xyz
OutputTiltFile	$FILENAME$.tlt
OutputXAxisTiltFile	$FILENAME$.xtilt
OutputTransformFile	$FILENAME$.tltxf
OutputFilledInModel     $FILENAME$_nogaps.fid
RotationAngle	$AXISANGLE$
TiltFile	$FILENAME$.rawtlt
#
# ADD a recommended tilt angle change to the existing AngleOffset value
#
AngleOffset	0.0
RotOption	0
RotDefaultGrouping	5
#
# TiltOption 0 fixes tilts, 2 solves for all tilt angles; change to 5 to solve
# for fewer tilts by grouping views by the amount in TiltDefaultGrouping
#
TiltOption	0
TiltDefaultGrouping	5
MagReferenceView	1
MagOption	0
MagDefaultGrouping	4
#
# To solve for distortion, change both XStretchOption and SkewOption to 3;
# to solve for skew only leave XStretchOption at 0
#
XStretchOption	0
SkewOption	0
XStretchDefaultGrouping	7
SkewDefaultGrouping	11
BeamTiltOption	0
#
# To solve for X axis tilt between two halves of a dataset, set XTiltOption to 4
#
XTiltOption	0
XTiltDefaultGrouping	2000
# 
# Criterion # of S.D's above mean residual to report (- for local mean)
#
ResidualReportCriterion	3.0
SurfacesToAnalyze	1
MetroFactor	0.25
MaximumCycles	1000
KFactorScaling	1.0
NoSeparateTiltGroups	1
#
# ADD a recommended amount to shift up to the existing AxisZShift value
#
AxisZShift	0.0
ShiftZFromOriginal      1
#
# Set to 1 to do local alignments
#
LocalAlignments	0
OutputLocalFile	$FILENAME$local.xf
#
# Target size of local patches to solve for in X and Y
#
TargetPatchSizeXandY	700,700
MinSizeOrOverlapXandY	0.5,0.5
#
# Minimum fiducials total and on one surface if two surfaces
#
MinFidsTotalAndEachSurface	8,3
FixXYZCoordinates	0
LocalOutputOptions	1,0,1
LocalRotOption	3
LocalRotDefaultGrouping	6
LocalTiltOption	5
LocalTiltDefaultGrouping	6
LocalMagReferenceView	1
LocalMagOption	3
LocalMagDefaultGrouping	7
LocalXStretchOption	0
LocalXStretchDefaultGrouping	7
LocalSkewOption	0
LocalSkewDefaultGrouping	11
RobustFitting	
WeightWholeTracks	
#
# COMBINE TILT TRANSFORMS WITH PREALIGNMENT TRANSFORMS
#
$xfproduct -StandardInput
InputFile1	$FILENAME$.prexg
InputFile2	$FILENAME$.tltxf
OutputFile	$FILENAME$_fid.xf
ScaleShifts	1.0,$BINNINGFACTOR$.0
$b3dcopy -p $FILENAME$_fid.xf $FILENAME$.xf
$b3dcopy -p $FILENAME$.tlt $FILENAME$_fid.tlt
#
# CONVERT RESIDUAL FILE TO MODEL
#
$if (-e $FILENAME$.resid) patch2imod -s 10 $FILENAME$.resid $FILENAME$.resmod
$if (-e ./savework) ./savework
