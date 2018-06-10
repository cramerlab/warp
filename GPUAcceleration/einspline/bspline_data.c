/////////////////////////////////////////////////////////////////////////////
//  einspline:  a library for creating and evaluating B-splines            //
//  Copyright (C) 2007 Kenneth P. Esler, Jr.                               //
//                                                                         //
//  This program is free software; you can redistribute it and/or modify   //
//  it under the terms of the GNU General Public License as published by   //
//  the Free Software Foundation; either version 2 of the License, or      //
//  (at your option) any later version.                                    //
//                                                                         //
//  This program is distributed in the hope that it will be useful,        //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of         //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          //
//  GNU General Public License for more details.                           //
//                                                                         //
//  You should have received a copy of the GNU General Public License      //
//  along with this program; if not, write to the Free Software            //
//  Foundation, Inc., 51 Franklin Street, Fifth Floor,                     //
//  Boston, MA  02110-1301  USA                                            //
/////////////////////////////////////////////////////////////////////////////

#include "config.h"

/*****************
/*   SSE Data    */
/*****************/

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif 

#define _XOPEN_SOURCE 600

#ifndef __USE_XOPEN2K
  #define __USE_XOPEN2K
#endif
#include <stdlib.h>


/*****************/
/* Standard Data */
/*****************/

//////////////////////
// Single precision //
//////////////////////
const float A44f[16] = 
  { -1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0,
     3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0,
    -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0,
     1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0 };
const float* restrict Af = A44f;

const float dA44f[16] =
  {  0.0, -0.5,  1.0, -0.5,
     0.0,  1.5, -2.0,  0.0,
     0.0, -1.5,  1.0,  0.5,
     0.0,  0.5,  0.0,  0.0 };
const float* restrict dAf = dA44f;

const float d2A44f[16] = 
  {  0.0, 0.0, -1.0,  1.0,
     0.0, 0.0,  3.0, -2.0,
     0.0, 0.0, -3.0,  1.0,
     0.0, 0.0,  1.0,  0.0 };
const float* restrict d2Af = d2A44f;