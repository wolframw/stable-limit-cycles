#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>
#include <math.h>

float clampf( float min, float max, float value );
int clampi( int min, int max, int value );
size_t clamps( size_t min, size_t max, size_t value );
float lerp( float a, float b, float t );

typedef struct
{
    float b0, b1, b2;
    float a1, a2;
} biquad_spec;

typedef struct
{
    float z1, z2;
} biquad_state;

#define butter8_4_size 4
extern const biquad_spec butter8_4[];

#define butter8_8_size 4
extern const biquad_spec butter8_8[];

#define butter8_12_size 4
extern const biquad_spec butter8_12[];

#define butter8_16_size 4
extern const biquad_spec butter8_16[];

#define butter8_20_size 4
extern const biquad_spec butter8_20[];

float biquad( float in, biquad_state *state, const biquad_spec *spec, size_t spec_size );

#endif
