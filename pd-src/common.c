#include "common.h"

const biquad_spec butter8_8[] = {
    { 8.88199322e-07,  1.77639864e-06,  8.88199322e-07,
     -1.34350206e+00,  4.54196154e-01 },
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.40173993e+00,  5.17232370e-01 },
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.52378987e+00,  6.49338274e-01 },
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.71939291e+00,  8.61057480e-01 }};

const biquad_spec butter8_12[] = {
    { 4.60202576e-08,  9.20405153e-08,  4.60202576e-08,
     -1.54074088e+00,  5.95092335e-01 },
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.58973945e+00,  6.45819386e-01 },
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.68898837e+00,  7.48569430e-01 },
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.83899511e+00,  9.03867829e-01 }};

const biquad_spec butter8_16[] = {
    { 5.34787079e-09,  1.06957416e-08,  5.34787079e-09,
     -1.64652218e+00,  6.78779458e-01 },
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.68779113e+00,  7.20856918e-01 },
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.76975340e+00,  8.04424922e-01 },
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.88965004e+00,  9.26670472e-01 }};

const biquad_spec butter8_20[] = {
    { 9.83559113e-10,  1.96711823e-09,  9.83559113e-10,
     -1.71261285e+00,  7.33960788e-01 }, 
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.74801189e+00,  7.69801081e-01 },
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.81742378e+00,  8.40078193e-01 },
    { 1.00000000e+00,  2.00000000e+00,  1.00000000e+00,
     -1.91687583e+00,  9.40769933e-01 }};

float clampf( float min, float max, float value )
{
    return value < min ? min : value > max ? max : value;
}

int clampi( int min, int max, int value )
{
    return value < min ? min : value > max ? max : value;
}

size_t clamps( size_t min, size_t max, size_t value )
{
    return value < min ? min : value > max ? max : value;
}

float lerp( float a, float b, float t )
{
    return a + t * (b - a);
}

float biquad( float in, biquad_state *state, const biquad_spec *spec, size_t spec_size )
{
    float x = in;
    size_t i;

    for ( i = 0; i < spec_size; ++i ) {
        float y     = x * spec[i].b0 + state[i].z1;
        state[i].z1 = x * spec[i].b1 + state[i].z2 - spec[i].a1 * y;
        state[i].z2 = x * spec[i].b2 - spec[i].a2 * y;
        x = y;
    }

    return x;
}
