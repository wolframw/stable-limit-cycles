#include "m_pd.h"
#include "common.h"
#include "lut.h"
#include <math.h>
#include <string.h>

static t_class *vdp_circle_tilde_class;

#define SUBSTEPS        24
#define OVERSAMPLING    12

typedef struct {
    /* PD object */
    t_object  obj;

    /* state */
    t_float      f;
    t_float      state[2];
    biquad_state filter_state[butter8_20_size];

    /* inlets */
    t_inlet  *inlet_freq;
    t_inlet  *inlet_mu;

    /* outlets */
    t_outlet *outlet;
} t_vdp_circle_tilde;

void *vdp_circle_tilde_new( void )
{
    t_vdp_circle_tilde *vdp_circle = (t_vdp_circle_tilde *) pd_new( vdp_circle_tilde_class );
    vdp_circle->state[0]    = 2.0f;
    vdp_circle->state[1]    = 0.0f;
    memset( vdp_circle->filter_state, 0, sizeof(float) * butter8_20_size );

    vdp_circle->inlet_freq  = inlet_new( &vdp_circle->obj, &vdp_circle->obj.ob_pd, &s_signal, &s_signal );
    vdp_circle->inlet_mu    = inlet_new( &vdp_circle->obj, &vdp_circle->obj.ob_pd, &s_signal, &s_signal );
    vdp_circle->outlet      = outlet_new( &vdp_circle->obj, &s_signal );

    return (void *) vdp_circle;
}

void vdp_circle_tilde_free( t_vdp_circle_tilde *vdp_circle )
{
    inlet_free( vdp_circle->inlet_freq );
    inlet_free( vdp_circle->inlet_mu );
    outlet_free( vdp_circle->outlet );
}

void vdp_circle( float *state, float mu )
{
    float tau = mu < 1.0f ? mu : 1.0f;

    float x  = state[0];
    float y  = state[1];
    
    float dxr = x * (2.0f / sqrt(x*x + y*y) - 1) + y;
    float dyr = y * (2.0f / sqrt(x*x + y*y) - 1) - x;

    float dx = lerp(dxr, y, tau);
    float dy = lerp(dyr, mu * (1 - x*x) * y - x, tau);

    state[0] = dx;
    state[1] = dy;
}

void solve( float *state, float mu, float dt )
{
    float k1[2] = { state[0], state[1] };
    vdp_circle( k1, mu );

    float k2[2] = {
        state[0] + (2.0f / 3) * dt * k1[0], 
        state[1] + (2.0f / 3) * dt * k1[1] };
    vdp_circle( k2, mu );

    state[0] += dt * (k1[0] + 3.0f * k2[0]) / 4.0f;
    state[1] += dt * (k1[1] + 3.0f * k2[1]) / 4.0f;
}

t_int *vdp_circle_tilde_perform( t_int *w )
{
    t_vdp_circle_tilde *vdp_circle      = (t_vdp_circle_tilde *)(w[1]);
    t_sample    *in_freq  = (t_sample *)(w[2]);
    t_sample    *in_mu    = (t_sample *)(w[3]);
    t_sample    *out      = (t_sample *)(w[4]);
    int          n        = (int)(w[5]);

    while ( n-- ) {
        float freq  = clampf( 0.0f, 2500.0f, *in_freq );
        float mu    = clampf( 0.0f, 10.0f, *in_mu );

        float sample = 0.0f;
        int i;
        for ( i = 0; i < SUBSTEPS; ++i ) {
            float period = vdp_circle_period( mu );
            float dt     = (period * freq) / (sys_getsr() * SUBSTEPS);

            solve( vdp_circle->state, mu, dt );

            if (i % (SUBSTEPS / OVERSAMPLING) == 0) {
                sample = (vdp_circle->state[0] / 2) * 0.828f;
                sample = biquad( sample, vdp_circle->filter_state, butter8_12, butter8_20_size );
            }
        }

        *out = sample;

        ++in_freq;
        ++in_mu;
        ++out;
    }

    return w + 6;
}

void vdp_circle_tilde_dsp( t_vdp_circle_tilde *vdp_circle, t_signal **sp )
{
    dsp_add( vdp_circle_tilde_perform, 5,
             vdp_circle,
             sp[0]->s_vec,  /* frequency */
             sp[1]->s_vec,  /* mu */
             sp[2]->s_vec,  /* out */
             sp[0]->s_n);
}

void vdp_circle_tilde_setup( void )
{
    vdp_circle_tilde_class = class_new(
        gensym("vdp_circle~"),
        (t_newmethod) vdp_circle_tilde_new,
        (t_method)    vdp_circle_tilde_free,
        sizeof(t_vdp_circle_tilde),
        CLASS_NOINLET,
        0
    );

    class_addmethod(
        vdp_circle_tilde_class,
        (t_method) vdp_circle_tilde_dsp,
        gensym("dsp"),
        A_CANT,
        0
    );

    CLASS_MAINSIGNALIN( vdp_circle_tilde_class, t_vdp_circle_tilde, f );
}
