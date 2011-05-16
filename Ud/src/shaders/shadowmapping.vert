uniform mat4 LightViewProjection;
varying vec4 ShadowTexCoord;

void main() 
{    
    ShadowTexCoord = LightViewProjection*gl_Vertex;    
    gl_Position = ftransform();
}
