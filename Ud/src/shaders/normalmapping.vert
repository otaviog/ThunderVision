#define LightsCount 1

varying vec3 vLight[4];
varying vec3 vHalf[4];
varying vec3 vEye;
varying vec3 pPosition;
attribute vec4 ud_Tangent;

void main()
{
    gl_Position = ftransform();
    vec3 position = vec3(gl_ModelViewMatrix*gl_Vertex);
    pPosition = position;

    vec3 vTangent = normalize(vec3(gl_NormalMatrix*ud_Tangent.xyz));
    vec3 vNormal = normalize(gl_NormalMatrix*gl_Normal);
    vec3 vBitangent = ud_Tangent.w*normalize(cross(vNormal, vTangent));      
    
    mat3 tangentSpace = mat3(vTangent.x, vBitangent.x, vNormal.x,
                             vTangent.y, vBitangent.y, vNormal.y,
                             vTangent.z, vBitangent.z, vNormal.z);

    gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;

    vEye = tangentSpace*normalize(-pPosition);

    vec3 tmpL[4], tmpH[4];
    for (int iLight=0; iLight<LightsCount; iLight++)
    {
        vec3 L, H;
    
        if ( gl_LightSource[iLight].position.w == 1.0 )
        {
            L = normalize(gl_LightSource[iLight].position.xyz - position);
            H = normalize(L + normalize(-position));
        }
        else 
        {
            L = gl_LightSource[iLight].position.xyz;
            H = gl_LightSource[iLight].halfVector.xyz;
        }
        
        L = tangentSpace*L;
        H = tangentSpace*H;

        // Fix for "Not supported when use temporary array indirect index." message on my ATI low-end hardware
        if ( iLight == 0 ) 
        {            
            vLight[0] = L;
            vHalf[0] = H;
        }
        else if ( iLight == 1 )            
        {
            vLight[1] = L;
            vHalf[1] = H;         
        }
        else if ( iLight == 2 )       
        {
            vLight[2] = L;
            vHalf[2] = H;         
        }
        else
        {
            vLight[3] = L;
            vHalf[3] = H;         
        }        
    }          
}

