uniform sampler2DShadow ud_DepthMap;
uniform float TexSize;

varying vec4 ShadowTexCoord;

/**
 * As done by Michael Bunnell and Fabio Pellacini, GPU Gems Vol I, Chapter 11. Shadow Map Antialiasing
 */
float pcf()
{
    float off = 1.0/TexSize;
    float sum = 0.0;

    for (float x = -1.5; x <= 1.5; x += 1.0)
        for (float y = -1.5; y <= 1.5; y += 1.0)
            sum += shadow2DProj(ud_DepthMap, vec4(ShadowTexCoord.x + x*off*ShadowTexCoord.w, 
                                                  ShadowTexCoord.y + y*off*ShadowTexCoord.w, 
                                                  ShadowTexCoord.z,
                                                  ShadowTexCoord.w)).w;
    return sum/16.0;
}

void main() 
{        
    float pcf = pcf();
    if ( pcf < 0.5f )
        gl_FragColor = vec4(0.0, 0.0, 0.0, (1-pcf) * 0.5);
    else
        gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);    
}
