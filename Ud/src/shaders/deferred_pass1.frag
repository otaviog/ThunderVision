#extension GL_ARB_draw_buffers : enable

varying vec3 pPosition;
varying vec3 vTangent;
varying vec3 vNormal;

void main()
{
    gl_FragData[0] = vec4(pPosition, 1);
    gl_FragData[1] = vec4(vNormal, 1);
    gl_FragData[2] = vec4(vTangent, 1);
    
    gl_FragData[3] = gl_FrontMaterial.ambient;
    gl_FragData[4] = gl_FrontMaterial.diffuse;
    gl_FragData[5] = gl_FrontMaterial.specular;
}
