#include "m_pd.h"
#include "common.h"
#include "lut.h"
#include <math.h>
#include <string.h>

static t_class *vdp_tilde_class;

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
} t_vdp_tilde;

void *vdp_tilde_new( void )
{
    t_vdp_tilde *vdp = (t_vdp_tilde *) pd_new( vdp_tilde_class );
    vdp->freq        = 0.0f;
    vdp->mu          = 0.0f;
    vdp->force       = 0.0f;
    vdp->state[0]    = 2.0f;
    vdp->state[1]    = 0.0f;
    memset( vdp->filter_state, 0, sizeof(float) * butter8_20_size );

    vdp->inlet_freq  = inlet_new( &vdp->obj, &vdp->obj.ob_pd, &s_signal, &s_signal );
    vdp->inlet_mu    = inlet_new( &vdp->obj, &vdp->obj.ob_pd, &s_signal, &s_signal );
    vdp->inlet_force = inlet_new( &vdp->obj, &vdp->obj.ob_pd, &s_signal, &s_signal );
    vdp->outlet      = outlet_new( &vdp->obj, &s_signal );

    return (void *) vdp;
}

void vdp_tilde_free( t_vdp_tilde *vdp )
{
    inlet_free( vdp->inlet_freq );
    inlet_free( vdp->inlet_mu );
    outlet_free( vdp->outlet );
}

void vdp( float *state, float mu, float force )
{
    float x   = state[0];
    float dx  = state[1];
    float ddx = mu * (1 - x * x) * dx - x + force;

    state[0] = dx;
    state[1] = ddx;
}

void solve( float *state, float mu, float force, float dt )
{
    float k1[2] = { state[0], state[1] };
    vdp( k1, mu, force );

    float k2[2] = {
        state[0] + (2.0f / 3) * dt * k1[0], 
        state[1] + (2.0f / 3) * dt * k1[1] };
    vdp( k2, mu, force );

    state[0] += dt * (k1[0] + 3.0f * k2[0]) / 4.0f;
    state[1] += dt * (k1[1] + 3.0f * k2[1]) / 4.0f;
}

t_int *vdp_tilde_perform( t_int *w )
{
    t_vdp_tilde *vdp      = (t_vdp_tilde *)(w[1]);
    t_sample    *in_freq  = (t_sample *)(w[2]);
    t_sample    *in_mu    = (t_sample *)(w[3]);
    t_sample    *in_force = (t_sample *)(w[4]);
    t_sample    *out      = (t_sample *)(w[5]);
    int          n        = (int)(w[6]);

    while ( n-- ) {
        float clamped_freq  = clampf( 0.0f, 2500.0f, *in_freq );
        float clamped_mu    = clampf( 0.0f, 10.0f, *in_mu );
        float clamped_force = clampf( -10.0f, 10.0f, *in_force );

        float sample = 0.0f;
        int i;
        for ( i = 0; i < OS_FACTOR; ++i ) {
            float freq   = lerp( clamped_freq, vdp->freq, (float) i / (OS_FACTOR - 1) );
            float mu     = lerp( clamped_mu, vdp->mu, (float) i / (OS_FACTOR - 1) );
            float force  = lerp( clamped_force, vdp->force, (float) i / (OS_FACTOR - 1) );
            float period = vdp_period( mu );
            float dt     = (period * freq) / (sys_getsr() * OS_FACTOR);

            solve( vdp->state, mu, force, dt );
            sample = vdp->state[0] / 2;
            sample = biquad( sample, vdp->filter_state, butter8_20, butter8_20_size );
        }

        vdp->freq = clamped_freq;
        vdp->mu = clamped_mu;

        *out = sample;

        ++in_freq;
        ++in_mu;
        ++in_force;
        ++out;
    }

    return w + 7;
}

void vdp_tilde_dsp( t_vdp_tilde *vdp, t_signal **sp )
{
    dsp_add( vdp_tilde_perform, 6,
             vdp,
             sp[0]->s_vec,  /* frequency */
             sp[1]->s_vec,  /* mu */
             sp[2]->s_vec,  /* force */
             sp[3]->s_vec,  /* out */
             sp[0]->s_n);
}

void vdp_tilde_setup( void )
{
    vdp_tilde_class = class_new(
        gensym("vdp~"),
        (t_newmethod) vdp_tilde_new,
        (t_method)    vdp_tilde_free,
        sizeof(t_vdp_tilde),
        CLASS_NOINLET,
        0
    );

    class_addmethod(
        vdp_tilde_class,
        (t_method) vdp_tilde_dsp,
        gensym("dsp"),
        A_CANT,
        0
    );

    CLASS_MAINSIGNALIN( vdp_tilde_class, t_vdp_tilde, f );
}
