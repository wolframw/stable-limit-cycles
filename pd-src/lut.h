#ifndef LUT_H
#define LUT_H

#define BRUSSEL_A_MIN   1.35f
#define BRUSSEL_A_MAX   2.2f

#define BRUSSEL_B_MIN   6.0f
#define BRUSSEL_B_MAX   8.0f

typedef struct
{
    float period;
    float gmin;
    float gmax;
} lut3;

float vdp_period( float mu );
float vdp_circle_period( float mu );
lut3 brussel_lookup( float a, float b );
lut3 lienard_even_lookup( float mu );

#endif