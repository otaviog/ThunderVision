uniform sampler2D ud_TexUnit0;
uniform sampler2D ud_TexUnit1;
uniform sampler2D ud_TexUnit2; // Enviroment Map
uniform sampler2D ud_TexUnit3; // specularmap
uniform sampler2D ud_TexUnit4; // Lightmap

#define LightsCount 1

varying vec3 vLight[4];
varying vec3 vHalf[4];
varying vec3 pPosition;
varying vec3 vEye;

vec4 ud_BlinnLight(int iLight, vec3 vNormal, vec3 vLightD, 
                   vec3 vHalf, vec3 pPosition, bool useSpecmap)
{    
    float fDiffuse = max(dot(vNormal, vLightD), 0.0);    
    vec4 colorSpecular = vec4(0, 0, 0, 0);
        
    if ( fDiffuse > 0.0 )
    {         
        float fSpecular = pow(max(dot(vNormal, vHalf), 0.0), gl_FrontMaterial.shininess);
        vec4 colorSpecular = gl_FrontMaterial.specular*gl_LightSource[iLight].specular*fSpecular;
        
        if ( useSpecmap )
            colorSpecular *= texture2D(ud_TexUnit3, gl_TexCoord[0].xy);
    }
     
    float d = length(gl_LightSource[iLight].position.xyz - pPosition);

    float attenuation = 1.0/
        (gl_LightSource[iLight].constantAttenuation
         + gl_LightSource[iLight].linearAttenuation*d
         + gl_LightSource[iLight].quadraticAttenuation*d*d);

    return (gl_FrontMaterial.ambient*gl_LightSource[iLight].ambient
            + gl_FrontMaterial.diffuse*gl_LightSource[iLight].diffuse*fDiffuse
            + colorSpecular
        )*attenuation;
}

void main()
{
    float h = texture2D(ud_TexUnit1, gl_TexCoord[0].xy).w*0.05;    
    vec2 offSet = vEye.xy*h;
//    offSet = offSet*0.05;

    vec3 vNormal = texture2D(ud_TexUnit1, gl_TexCoord[0].xy + offSet).xyz;
    vNormal = (vNormal - 0.5)*2.0;
        
    vec4 cFragFinal = vec4(0, 0, 0, 0); 
    
    for (int iLight=0; iLight<LightsCount; iLight++)
    {
        cFragFinal += ud_BlinnLight(iLight, vNormal, vLight[iLight],
                                    vHalf[iLight], pPosition, true);
    }
    
    
    cFragFinal /= float(LightsCount);   
    vec4 texContrib = texture2D(ud_TexUnit0, gl_TexCoord[0].xy + offSet);  

    gl_FragColor = cFragFinal*texContrib;
}
