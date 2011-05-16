#version 120
#extension GL_EXT_geometry_shader4 : enable

//uniform int Light;

vec4 extrude(vec4 position)
{
    return vec4(position.xyz + (position.xyz - gl_LightSource[0].position.xyz)*500.0, 1.0);
}

void emitEdgeExtrusion(vec4 a, vec4 b)
{
    gl_Position = gl_ProjectionMatrix*a;
    EmitVertex();
    gl_Position = gl_ProjectionMatrix*extrude(a);
    EmitVertex();
    gl_Position = gl_ProjectionMatrix*b;    
    EmitVertex();        
    gl_Position = gl_ProjectionMatrix*extrude(b);
    EmitVertex();
    EndPrimitive();
}

void main()
{
    vec4 t0_a = gl_PositionIn[0];
    vec4 t0_b = gl_PositionIn[2];
    vec4 t0_c = gl_PositionIn[4];
    
    vec4 a = gl_PositionIn[1];
    vec4 b = gl_PositionIn[3];
    vec4 c = gl_PositionIn[5];
    
    vec3 t0_norm = normalize(cross(t0_b.xyz - t0_a.xyz, t0_c.xyz - t0_a.xyz));        
    float d0 = dot(t0_norm, normalize(gl_LightSource[0].position.xyz - t0_a.xyz));
        
    if ( d0 > 0.0f )
    {
#if 0
        vec3 t1_norm = normalize(cross(t0_a.xyz - a.xyz, t0_b.xyz - a.xyz));    
        vec3 t2_norm = normalize(cross(t0_b.xyz - b.xyz, t0_c.xyz - b.xyz));    
        vec3 t3_norm = normalize(cross(t0_c.xyz - c.xyz, t0_a.xyz - c.xyz));
#else 
        // Best
        vec3 t1_norm = normalize(cross(t0_b.xyz - a.xyz, t0_a.xyz - a.xyz));
        vec3 t2_norm = normalize(cross(t0_c.xyz - b.xyz, t0_b.xyz - b.xyz));
        vec3 t3_norm = normalize(cross(t0_a.xyz - c.xyz, t0_c.xyz - c.xyz));

#endif
        float d1 = dot(t1_norm, normalize(gl_LightSource[0].position.xyz - a.xyz));
        float d2 = dot(t2_norm, normalize(gl_LightSource[0].position.xyz - b.xyz));
        float d3 = dot(t3_norm, normalize(gl_LightSource[0].position.xyz - c.xyz));

#if 0 
        if ( d1 < 0.0 )// || !(d1 == d1) )
        {
            emitEdgeExtrusion(t0_a, t0_b);
        }
    
        if ( d2 < 0.0 )//|| !(d2 == d2) )
        {
            emitEdgeExtrusion(t0_b, t0_c);        
        }

        if ( d3 < 0.0 ) // || !(d3 == d3) )
        {
            emitEdgeExtrusion(t0_c, t0_a);
        }
#else
        if ( d1 < 0.0 || !(d1 == d1) )
        {
            emitEdgeExtrusion(t0_a, t0_b);
        }
    
        if ( d2 < 0.0 || !(d2 == d2) )
        {
            emitEdgeExtrusion(t0_b, t0_c);        
        }

        if ( d3 < 0.0 || !(d3 == d3) )
        {
            emitEdgeExtrusion(t0_c, t0_a);
        }

#endif
    }
}
