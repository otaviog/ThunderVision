uniform mat4 ViewProjectionCropMatrix;
uniform vec2 WaterMove;

varying vec3 vLight;
varying vec3 vHalf;
varying vec3 pPosition;

void main() 
{
    gl_Position = ftransform();
    gl_TexCoord[0] = vec4(gl_MultiTexCoord0.xy + WaterMove, 0, 0);
    gl_TexCoord[1] = gl_Position;

    vec3 position = vec3(gl_ModelViewMatrix*gl_Vertex);
    pPosition = position;
    
    vec3 vTangent = gl_NormalMatrix*vec3(1, 0, 0);
    vec3 vNormal = gl_NormalMatrix*vec3(0, 1, 0); 
    vec3 vBinormal = normalize(cross(vNormal, vTangent));
    
    mat3 tangentSpace = mat3(vTangent.x, vBinormal.x, vNormal.x,
                             vTangent.y, vBinormal.y, vNormal.y,
                             vTangent.z, vBinormal.z, vNormal.z);
    
    vec3 L = vec3(gl_LightSource[0].position.xyz);
    L = normalize(L - position);

    vec3 H = normalize(L + normalize(-position));

    vLight = tangentSpace*L;
    vHalf = tangentSpace*H;    
}
