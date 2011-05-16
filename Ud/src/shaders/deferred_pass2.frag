#extension GL_ARB_texture_rectangle : enable

uniform sampler2DRect positionTex;
uniform sampler2DRect normalTex;
uniform sampler2DRect tangentTex;
uniform sampler2DRect ambientTex;
uniform sampler2DRect diffuseTex;
uniform sampler2DRect specularTex;
uniform int lightCount;

vec4 ud_BlinnLight(int iLight, vec3 vNormal, vec3 vLightD, 
                   vec3 vHalf, vec3 pPosition,
                   vec4 ambient, vec4 diffuse, vec4 specular)
{    
    float fDiffuse = max(dot(vNormal, vLightD), 0.0);    
    vec4 colorSpecular = vec4(0, 0, 0, 0);
        
    if ( fDiffuse > 0.0 )
    {         
        float fSpecular = pow(max(dot(vNormal, vHalf), 0.0), gl_FrontMaterial.shininess);
        vec4 colorSpecular = specular*gl_LightSource[iLight].specular*fSpecular;        
    }
     
    float d = length(gl_LightSource[iLight].position.xyz - pPosition);

    float attenuation = 1.0/
        (gl_LightSource[iLight].constantAttenuation
         + gl_LightSource[iLight].linearAttenuation*d
         + gl_LightSource[iLight].quadraticAttenuation*d*d);

    return (ambient*gl_LightSource[iLight].ambient
            + diffuse*gl_LightSource[iLight].diffuse*fDiffuse
            + colorSpecular
        )*attenuation;
}

void main()
{
    vec3 pos = texture2DRect(positionTex, gl_FragCoord.xy).xyz;
    vec3 normal = texture2DRect(normalTex, gl_FragCoord.xy).xyz;
    vec3 tangent = texture2DRect(tangentTex, gl_FragCoord.xy).xyz;
    
    vec3 ambient = texture2DRect(ambientTex, gl_FragCoord.xy).xyz;
    vec3 diffuse = texture2DRect(diffuseTex, gl_FragCoord.xy).xyz;
    vec3 specular = texture2DRect(specularTex, gl_FragCoord.xy).xyz;
    vec4 fragColor = vec4(0.0, 0.0, 0.0, 0.0);
    
    for (int i=0; i<lightCount; i++)
    {
        vec3 vLight = normalize(gl_LightSource[i].position.xyz - pos);
        vec3 vHalf = normalize(vLight + normalize(-pos));
    
        fragColor += ud_BlinnLight(i, normal, vLight, vHalf, pos,
                                   vec4(ambient, 1), vec4(diffuse, 1), 
                                   vec4(specular, 1));
    }

    gl_FragColor = fragColor/float(lightCount);
}
