varying vec3 pPosition;
varying vec3 vTangent;
varying vec3 vNormal;

attribute vec4 ud_Tangent;

void main()
{
    pPosition = vec3(gl_ModelViewMatrix*gl_Vertex);
    vNormal = gl_NormalMatrix*gl_Normal;
    vTangent = gl_NormalMatrix*ud_Tangent.xyz;
    
    gl_Position = ftransform();    
}
