#include "m_pd.h"
#include "common.h"
#include "lut.h"
#include <math.h>
#include <string.h>

static t_class *lienard_even_tilde_class;

#define OS_FACTOR 20

typedef struct {
    /* PD object */
    t_object  obj;

    /* state */
    t_float      f;
    t_float      freq;
    t_float      mu;
    t_float      force;
    t_float      state[2];
    biquad_state filter_state[butter8_20_size];

    /* inlets */
    t_inlet  *inlet_freq;
    t_inlet  *inlet_mu;
    t_inlet  *inlet_force;

    /* outlets */
    t_outlet *outlet;
} t_lienard_even_tilde;

void *lienard_even_tilde_new( void )
{
    t_lienard_even_tilde *lienard_even = (t_lienard_even_tilde *) pd_new( lienard_even_tilde_class );
    lienard_even->freq        = 0.0f;
    lienard_even->mu          = 0.0f;
    lienard_even->force       = 0.0f;
    lienard_even->state[0]    = 2.0f;
    lienard_even->state[1]    = 0.0f;
    memset( lienard_even->filter_state, 0, sizeof(float) * butter8_20_size );

    lienard_even->inlet_freq  = inlet_new( &lienard_even->obj, &lienard_even->obj.ob_pd, &s_signal, &s_signal );
    lienard_even->inlet_mu    = inlet_new( &lienard_even->obj, &lienard_even->obj.ob_pd, &s_signal, &s_signal );
    lienard_even->inlet_force = inlet_new( &lienard_even->obj, &lienard_even->obj.ob_pd, &s_signal, &s_signal );
    lienard_even->outlet      = outlet_new( &lienard_even->obj, &s_signal );

    return (void *) lienard_even;
}

void lienard_even_tilde_free( t_lienard_even_tilde *lienard_even )
{
    inlet_free( lienard_even->inlet_freq );
    inlet_free( lienard_even->inlet_mu );
    outlet_free( lienard_even->outlet );
}

void lienard_even( float *state, float mu, float force )
{
    float x   = state[0];
    float dx  = state[1];
    float ddx = mu * (1 + x - (x*x)) * dx - (exp( x ) - 1) + force;

    state[0] = dx;
    state[1] = ddx;
}

void solve( float *state, float mu, float force, float dt )
{
    float k1[2] = { state[0], state[1] };
    lienard_even( k1, mu, force );

    float k2[2] = {
        state[0] + (2.0f / 3) * dt * k1[0], 
        state[1] + (2.0f / 3) * dt * k1[1] };
    lienard_even( k2, mu, force );

    state[0] += dt * (k1[0] + 3.0f * k2[0]) / 4.0f;
    state[1] += dt * (k1[1] + 3.0f * k2[1]) / 4.0f;
}

t_int *lienard_even_tilde_perform( t_int *w )
{
    t_lienard_even_tilde *lienard_even = (t_lienard_even_tilde *)(w[1]);
    t_sample    *in_freq  = (t_sample *)(w[2]);
    t_sample    *in_mu    = (t_sample *)(w[3]);
    t_sample    *in_force = (t_sample *)(w[4]);
    t_sample    *out      = (t_sample *)(w[5]);
    int          n        = (int)(w[6]);

    while ( n-- ) {
        float clamped_freq  = clampf( 0.0f, 4000.0f, *in_freq );
        float clamped_mu    = clampf( 0.0f, 5.0f, *in_mu );
        float clamped_force = clampf( -10.0f, 10.0f, *in_force );

        float sample = 0.0f;
        int i;
        for ( i = 0; i < OS_FACTOR; ++i ) {
            float freq   = lerp( clamped_freq, lienard_even->freq, (float) i / (OS_FACTOR - 1) );
            float mu     = lerp( clamped_mu, lienard_even->mu, (float) i / (OS_FACTOR - 1) );
            float force  = lerp( clamped_force, lienard_even->force, (float) i / (OS_FACTOR - 1) );
            lut3 lc = lienard_even_lookup( mu );
            float dt     = (lc.period * freq) / (sys_getsr() * OS_FACTOR);

            solve( lienard_even->state, mu, force, dt );
            sample = (0.9 / (lc.gmin - lc.gmax)) * (lc.gmax + lc.gmin - 2 * lienard_even->state[0]);
            sample = biquad( sample, lienard_even->filter_state, butter8_20, butter8_20_size );
        }

        lienard_even->freq = clamped_freq;
        lienard_even->mu = clamped_mu;

        *out = sample;

        ++in_freq;
        ++in_mu;
        ++in_force;
        ++out;
    }

    return w + 7;
}

void lienard_even_tilde_dsp( t_lienard_even_tilde *lienard_even, t_signal **sp )
{
    dsp_add( lienard_even_tilde_perform, 6,
             lienard_even,
             sp[0]->s_vec,  /* frequency */
             sp[1]->s_vec,  /* mu */
             sp[2]->s_vec,  /* force */
             sp[3]->s_vec,  /* out */
             sp[0]->s_n);
}

void lienard_even_tilde_setup( void )
{
    lienard_even_tilde_class = class_new(
        gensym("lienard_even~"),
        (t_newmethod) lienard_even_tilde_new,
        (t_method)    lienard_even_tilde_free,
        sizeof(t_lienard_even_tilde),
        CLASS_NOINLET,
        0
    );

    class_addmethod(
        lienard_even_tilde_class,
        (t_method) lienard_even_tilde_dsp,
        gensym("dsp"),
        A_CANT,
        0
    );

    CLASS_MAINSIGNALIN( lienard_even_tilde_class, t_lienard_even_tilde, f );
}
