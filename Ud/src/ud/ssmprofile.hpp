#ifndef HG_SSMPROFILE_HPP
#define HG_SSMPROFILE_HPP

#include "common.hpp"
#include "renderprofile.hpp"
#include "matrix44.hpp"
#include "rendertexture.hpp"
#include "shader.hpp"
#include "projection.hpp"

UD_NAMESPACE_BEGIN

/**
 * Implements Standard Shadow Mapping rendering profile. It's three passe
 * technique. One to render the scene with lights, other to render the scene 
 * from light the light view and a final to draw the shadowned scene.
 * This class also uses the shadowmapping.* shaders.
 * 
 * @author Otávio Gomes
 */
class SSMProfile: public IRenderProfile
{
public:
    /**
     * Initializes the shader and texture need to draw cycle. 
     * 
     * @param lightProjection light projection parameters
     * @param shadowMapWidth the width of the shadow map, need be multiply of two
     * @param shadowMapHeight the height of the shadow map, need be multiply of two
     * @throws ud::ShaderException if one of the shadowmapping.* can't be load.     
     */
    SSMProfile(const Projection &lightProjection,
               int shadowMapWidth = 512,
               int shadowMapHeight = 512);

    /**
     * Renders with shadowing.
     *
     * @param camera defines the view transformation
     * @param projection defines the projection transformation
     * @param world contains the scene to draw
     */
    void draw(const Camera &camera, const Projection &projection, World *world);

private:
    RenderTexture m_framebuffer;
    ShaderProgram m_shadowPassShader;
    Texture m_depthMap;
    Projection m_lightProjection;
    int m_shadowMapWidth;
};

UD_NAMESPACE_END

#endif
