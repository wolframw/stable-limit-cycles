#include "m_pd.h"
#include "common.h"
#include "lut.h"
#include <math.h>
#include <string.h>

static t_class *brussel_tilde_class;

#define SUBSTEPS        24
#define OVERSAMPLING    12

typedef struct {
    /* PD object */
    t_object  obj;

    /* state */
    t_float      f;
    t_float      freq;
    t_float      a;
    t_float      b;
    t_float      state[2];
    biquad_state filter_state[butter8_20_size];

    /* inlets */
    t_inlet  *inlet_freq;
    t_inlet  *inlet_a;
    t_inlet  *inlet_b;

    /* outlets */
    t_outlet *outlet;
} t_brussel_tilde;

void *brussel_tilde_new( void )
{
    t_brussel_tilde *brussel = (t_brussel_tilde *) pd_new( brussel_tilde_class );
    brussel->freq        = 0.0f;
    brussel->a           = BRUSSEL_A_MAX;
    brussel->b           = BRUSSEL_B_MIN;
    brussel->state[0]    = 1.8f;
    brussel->state[1]    = 2.8f;
    memset( brussel->filter_state, 0, sizeof(float) * butter8_20_size );

    brussel->inlet_freq = inlet_new( &brussel->obj, &brussel->obj.ob_pd, &s_signal, &s_signal );
    brussel->inlet_a    = inlet_new( &brussel->obj, &brussel->obj.ob_pd, &s_signal, &s_signal );
    brussel->inlet_b    = inlet_new( &brussel->obj, &brussel->obj.ob_pd, &s_signal, &s_signal );
    brussel->outlet     = outlet_new( &brussel->obj, &s_signal );

    return (void *) brussel;
}

void brussel_tilde_free( t_brussel_tilde *brussel )
{
    inlet_free( brussel->inlet_freq );
    inlet_free( brussel->inlet_a );
    inlet_free( brussel->inlet_b );
    outlet_free( brussel->outlet );
}

void brussel( float *state, float a, float b )
{
    float x  = state[0];
    float y  = state[1];
    float dx = a + (x * x) * y - b * x - x;
    float dy = b * x - (x * x) * y;

    state[0] = dx;
    state[1] = dy;
}

void solve( float *state, float a, float b, float dt )
{
    float k1[2] = { state[0], state[1] };
    brussel( k1, a, b );

    float k2[2] = {
        state[0] + (2.0f / 3) * dt * k1[0], 
        state[1] + (2.0f / 3) * dt * k1[1] };
    brussel( k2, a, b );

    state[0] += dt * (k1[0] + 3.0f * k2[0]) / 4.0f;
    state[1] += dt * (k1[1] + 3.0f * k2[1]) / 4.0f;
}

t_int *brussel_tilde_perform( t_int *w )
{
    t_brussel_tilde *brussel = (t_brussel_tilde *)(w[1]);
    t_sample    *in_freq = (t_sample *)(w[2]);
    t_sample    *in_a    = (t_sample *)(w[3]);
    t_sample    *in_b    = (t_sample *)(w[4]);
    t_sample    *out     = (t_sample *)(w[5]);
    int          n       = (int)(w[6]);

    while ( n-- ) {
        float freq = clampf( 20.0f, 1000.0f, *in_freq );
        float a    = clampf( BRUSSEL_A_MIN, BRUSSEL_A_MAX, *in_a );
        float b    = clampf( BRUSSEL_B_MIN, BRUSSEL_B_MAX, *in_b );

        float sample = 0.0f;
        int i;
        for ( i = 0; i < SUBSTEPS; ++i ) {
            lut3  lc = brussel_lookup( a, b );
            float dt = (lc.period * freq) / (sys_getsr() * SUBSTEPS);
            solve( brussel->state, a, b, dt );

            if (i % (SUBSTEPS / OVERSAMPLING) == 0) {
                /* 0.83 is the fudge factor here because with a low, b high, things clip quite a bit */
                sample = (0.83f / (lc.gmin - lc.gmax)) * (lc.gmax + lc.gmin - 2 * brussel->state[0]);
                sample = biquad( sample, brussel->filter_state, butter8_12, butter8_20_size );
            }
        }

        *out = sample;

        ++in_freq;
        ++in_a;
        ++in_b;
        ++out;
    }

    return w + 7;
}

void brussel_tilde_dsp( t_brussel_tilde *brussel, t_signal **sp )
{
    dsp_add( brussel_tilde_perform, 6,
             brussel,
             sp[0]->s_vec,  /* frequency */
             sp[1]->s_vec,  /* a */
             sp[2]->s_vec,  /* b */
             sp[3]->s_vec,  /* out */
             sp[0]->s_n);
}

void brussel_tilde_setup( void )
{
    brussel_tilde_class = class_new(
        gensym("brussel~"),
        (t_newmethod) brussel_tilde_new,
        (t_method)    brussel_tilde_free,
        sizeof(t_brussel_tilde),
        CLASS_NOINLET,
        0
    );

    class_addmethod(
        brussel_tilde_class,
        (t_method) brussel_tilde_dsp,
        gensym("dsp"),
        A_CANT,
        0
    );

    CLASS_MAINSIGNALIN( brussel_tilde_class, t_brussel_tilde, f );
}
