AM_LDFLAGS = $(ALL_STATIC)

if WANT_FORTRAN
  MAYBE_FORTRAN = fbspline.c        \
                  fbspline.h        \
                  fmulti_bspline.c  \
                  fmulti_bspline.h  \
	          fnubspline.c      \
                  fnubspline.h

endif

if WANT_BLIPS
   MAYBE_BLIPS = blip_create.c blip_create.h
   CHECK_BLIPS = test_blip
endif

if HAVE_CUDA
   MAYBE_CUDA = multi_bspline_create_cuda.cu \
                multi_bspline_cuda_s_impl.h  \
                multi_bspline_cuda_c_impl.h  \
                multi_bspline_cuda_d_impl.h  \
                multi_bspline_cuda_z_impl.h  \
                multi_bspline_eval_cuda.h    \
                bspline_structs_cuda.h       \
                bspline_create_cuda.cu
   CHECK_CUDA = test_multi_cuda
   test_multi_cuda_SOURCES = test_multi_cuda.cu
   test_multi_cuda_LDADD   = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIB) $(CUDA_LIBS)
endif

if HAVE_SSE
   SINGLE_SRC = multi_bspline_eval_sse_s.c multi_bspline_eval_sse_s_cpp.cc       \
                multi_bspline_eval_sse_c.c multi_bspline_eval_sse_c_cpp.cc       \
                multi_bspline_eval_sse_s_impl.h  multi_bspline_eval_sse_c_impl.h 
else
   SINGLE_SRC = multi_bspline_eval_std_s.c multi_bspline_eval_std_s_cpp.cc       \
                multi_bspline_eval_std_c.c multi_bspline_eval_std_c_cpp.cc       \
                multi_bspline_eval_std_s_impl.h  multi_bspline_eval_std_c_impl.h 
endif

if HAVE_SSE2
   DOUBLE_SRC = multi_bspline_eval_sse_d.c multi_bspline_eval_sse_d_cpp.cc       \
                multi_bspline_eval_sse_z.c multi_bspline_eval_sse_z_cpp.cc       \
                multi_bspline_eval_sse_d_impl.h multi_bspline_eval_sse_z_impl.h  \
                multi_nubspline_eval_sse_z.c multi_nubspline_eval_sse_z_cpp.cc   \
                multi_nubspline_eval_sse_z_impl.h
else
   DOUBLE_SRC = multi_bspline_eval_std_d.c multi_bspline_eval_std_d_cpp.cc       \
                multi_bspline_eval_std_z.c multi_bspline_eval_std_z_cpp.cc       \
                multi_bspline_eval_std_d_impl.h multi_bspline_eval_std_z_impl.h  \
                multi_nubspline_eval_std_z.c multi_nubspline_eval_std_z_cpp.cc   \
                multi_nubspline_eval_std_z_impl.h

endif

check_PROGRAMS =  TestBspline TestNUBspline test_fbspline \
                  test_bspline_s test_bspline_d test_multi \
                  test_fmulti_bspline test_multi_cpp $(CHECK_BLIPS) 

bin_PROGRAMS = time_multi $(CHECK_CUDA)


TestBspline_SOURCES         = TestBspline.c
TestNUBspline_SOURCES       = TestNUBspline.c
test_fbspline_SOURCES       = test_fbspline.f
test_fmulti_bspline_SOURCES = test_fmulti_bspline.f
if WANT_BLIPS
   test_blip_SOURCES     = test_blip.c
endif
test_bspline_s_SOURCES = test_bspline_s.c
test_bspline_d_SOURCES = test_bspline_d.c
test_multi_SOURCES     = test_multi.c
test_multi_cpp_SOURCES = test_multi_cpp.cc
time_multi_SOURCES     = time_multi.c

LDADD = 
AM_CCFLAGS = -g 

TestBspline_LDADD    = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
TestNUBspline_LDADD  = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_fbspline_LDADD  = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_fmulti_bspline_LDADD = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_blip_LDADD      = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_bspline_s_LDADD = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_bspline_d_LDADD = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_multi_LDADD     = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_multi_cpp_LDADD = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
time_multi_LDADD     = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)


lib_LTLIBRARIES = libeinspline.la
libeinspline_la_SOURCES =                             \
                       aligned_alloc.h                \
                       bspline_base.h                 \
                       bspline_create.c               \
                       bspline_create.h               \
                       bspline_structs.h              \
                       bspline_data.c                 \
                       bspline_eval_std_s.h           \
                       bspline_eval_sse_s.h           \
                       bspline_eval_std_c.h           \
                       bspline_eval_sse_c.h           \
                       bspline_eval_std_d.h           \
                       bspline_eval_sse_d.h           \
                       bspline_eval_std_z.h           \
                       bspline_eval_sse_z.h           \
                       multi_bspline.h                \
                       multi_bspline_create.c         \
                       multi_bspline_create.h         \
                       multi_bspline_eval_c.h         \
                       multi_bspline_eval_d.h         \
                       multi_bspline_eval_s.h         \
                       multi_bspline_eval_z.h         \
                       multi_bspline_structs.h        \
                       multi_nubspline_create.c       \
                       nubspline_base.h               \
                       nubspline_create.c             \
                       nubspline_create.h             \
                       nubspline_eval_sse_s.h         \
                       nubspline_eval_std_s.h         \
                       nubspline_eval_sse_c.h         \
                       nubspline_eval_std_c.h         \
                       nubspline_eval_sse_d.h         \
                       nubspline_eval_std_d.h         \
                       nubspline_eval_sse_z.h         \
                       nubspline_eval_std_z.h         \
                       nubspline_structs.h            \
                       nubasis.h                      \
                       nubasis.c                      \
                       nugrid.h                       \
                       nugrid.c                       \
                       $(SINGLE_SRC)                  \
                       $(DOUBLE_SRC)                  \
                       $(MAYBE_BLIPS)                 \
                       $(MAYBE_CUDA)                  \
                       $(MAYBE_FORTRAN) 


libeinspline_la_LIBADD = $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)


includedir=$(prefix)/include/einspline

.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $<

.cu.lo:
	$(top_builddir)/cudalt.py $@ $(NVCC) -c $(NVCCFLAGS) $<

include_HEADERS = bspline_base.h bspline_create.h bspline_eval_sse_c.h        \
                  bspline_eval_sse_d.h bspline_eval_sse_s.h                   \
                  bspline_eval_sse_z.h bspline_eval_std_c.h                   \
                  bspline_eval_std_d.h bspline_eval_std_s.h                   \
                  bspline_eval_std_z.h bspline.h bspline_structs.h            \
                  fbspline.h fmulti_bspline.h  fnubspline.h                   \
                  multi_bspline.h                                             \
                  multi_bspline_create.h    multi_bspline_structs.h           \
                  multi_bspline_eval_c.h    multi_bspline_eval_d.h            \
                  multi_bspline_eval_s.h    multi_bspline_eval_z.h            \
                  multi_nubspline.h                                           \
                  multi_nubspline_create.h    multi_nubspline_structs.h       \
                  multi_nubspline_eval_c.h    multi_nubspline_eval_d.h        \
                  multi_nubspline_eval_s.h    multi_nubspline_eval_z.h        \
                  nubspline_base.h nubspline_create.h                         \
                  nubspline_eval_sse_s.h nubspline_eval_std_s.h               \
	          nubspline_eval_sse_c.h nubspline_eval_std_c.h               \
                  nubspline_eval_sse_d.h nubspline_eval_std_d.h               \
	          nubspline_eval_sse_z.h nubspline_eval_std_z.h               \
                  nubspline_structs.h nubasis.h nubspline.h nugrid.h          \
                  multi_bspline_structs_cuda.h multi_bspline_create_cuda.h    \
                  multi_bspline_eval_cuda.h bspline_eval_cuda.h               \
		  bspline_structs_cuda.h bspline_create_cuda.h                \
                  config.h

EXTRA_DIST = multi_bspline_eval_cuda_c.cu
