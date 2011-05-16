uniform sampler2D ud_TexUnit0; // DUDVmap
uniform sampler2D ud_TexUnit1; // Normalmap
uniform sampler2D ud_TexUnit2; // Environment cubemap
uniform sampler2D ud_TexUnit3; // Reflection
uniform sampler2D ud_TexUnit4; // Refraction 

varying vec3 vLight;
varying vec3 vHalf;
varying vec3 pPosition;

 vec4 ud_BlinnLight(int iLight, vec3 vNormal, vec3 vLightD, 
                    vec3 vHalf, vec3 pPosition)
 {    
     float fDiffuse = max(dot(vNormal, vLightD), 0.0);
     float fSpecular = 0.0;

     if ( fDiffuse > 0.0 )
         fSpecular = pow(max(dot(vNormal, vHalf), 0.0), gl_FrontMaterial.shininess);

     float d = length(gl_LightSource[iLight].position.xyz - pPosition);

     float attenuation = 1.0/
       (gl_LightSource[iLight].constantAttenuation
       + gl_LightSource[iLight].linearAttenuation*d
       + gl_LightSource[iLight].quadraticAttenuation*d*d);

     return (gl_FrontMaterial.ambient*gl_LightSource[iLight].ambient
             + gl_FrontMaterial.diffuse*gl_LightSource[iLight].diffuse*fDiffuse
             + gl_FrontMaterial.specular*gl_LightSource[iLight].specular*fSpecular
         )*attenuation;
 }

 void main()
 {        
     vec3 vDudvNormal = (texture2D(ud_TexUnit0, gl_TexCoord[0].xy).xyz - 0.5)*2.0;
     vec3 vNormal = normalize(texture2D(ud_TexUnit1, gl_TexCoord[0].xy).xyz*2.0 - 1.0);
     vec4 txproj = gl_TexCoord[1] + vec4(vDudvNormal, 0);
     txproj = txproj/txproj.w;
     txproj = txproj*0.5 + 0.5;

     vec4 reflecSamp = texture2DProj(ud_TexUnit3, txproj);
     vec4 refracSamp = texture2DProj(ud_TexUnit4, txproj);

     gl_FragColor = reflecSamp*refracSamp*
         ud_BlinnLight(
         0, normalize(vNormal), normalize(vLight), 
         vHalf, pPosition);         
}
