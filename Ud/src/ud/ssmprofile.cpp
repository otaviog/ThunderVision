#include <limits>
#include <iostream>
#include "frustum.hpp"
#include "concretenode.hpp"
#include "world.hpp"
#include "scene.hpp"
#include "debug.hpp"
#include "depthmap.hpp"
#include "camera.hpp"
#include "glstate.hpp"
#include "waterplane.hpp"
#include "skybox.hpp"
#include "ssmprofile.hpp"

UD_NAMESPACE_BEGIN

SSMProfile::SSMProfile(const Projection &lightProjection,
                       int shadowMapWidth,
                       int shadowMapHeight)
    : m_framebuffer(shadowMapWidth, shadowMapHeight),
      m_lightProjection(lightProjection)
{
    m_shadowPassShader.load("shadowmapping.vert", "shadowmapping.frag");
    m_depthMap.parameters(TextureParameters::ShadowMap());
    m_shadowMapWidth = shadowMapWidth;
}

void SSMProfile::draw(const Camera &camera, const Projection &projection, World *pWorld)
{
    Matrix44f matView;
    camera.genMatrix(&matView);

    Scene scene(pWorld, Frustum(matView, projection));
    scene.updateAnimated();
    scene.sortTransparents();
    projection.applyGL();
    
    glPushAttrib(GL_POLYGON_BIT);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);
    glLoadIdentity();
    camera.lookAt();
    
    for (int i = 0; i < pWorld->getLightCount(); ++i)
        pWorld->getLight(i).applyGL(GL_LIGHT0+i);

    VBOMesh::GLState renderState_1;
    renderState_1.push();

    for (Scene::RendableList::iterator it = scene.elements.begin();
         it != scene.elements.end() ; ++it)
    {
        Rendable *elem = *it;
        glPushMatrix();
        glMultMatrixf(elem->matrix().m);
        elem->mesh()->render();
        glPopMatrix();
    }

    renderState_1.pop();
      
    /**
     * Render with shadowing
     **/
    
    glstate::DisableMaterial stateDisMat_1;
    stateDisMat_1.push();

    for (int i = 0; i < pWorld->getLightCount(); ++i)
    {
        const Light &light = pWorld->getLight(i);

        Matrix44f matLightView;
        Matrix44f matLightProjection;
        
        light.genViewMatrix(&matLightView);
        m_lightProjection.genMatrix(&matLightProjection);
        
        glPolygonOffset(1.0f, 0.0f);
        depthmap::RenderDepthMap(m_framebuffer, m_depthMap.oglName(),
                                 matLightView, matLightProjection,
                                 pWorld);

        VBOMesh::GLStateNoMaterial renderNoMateState_2;
        renderNoMateState_2.push(); 
        
        glstate::Transparency stateTransp_3;
        stateTransp_3.push();
        
        glDepthMask(GL_FALSE);
        m_shadowPassShader.enable();
        
        glBindTexture(GL_TEXTURE_2D, m_depthMap.oglName());
        m_shadowPassShader.uniform("ud_DepthMap").seti(0);
        m_shadowPassShader.uniform("TexSize").setf(m_shadowMapWidth);

        glPolygonOffset(-1.0f, -1.0f);
        
        for (Scene::RendableList::iterator it = scene.elements.begin();
             it != scene.elements.end() ; ++it)
        {
            Rendable *elem = *it;            
            m_shadowPassShader.uniform("LightViewProjection").set(
                Matrix44f::TextureProjectionCrop() 
                * matLightProjection 
                * matLightView * elem->matrix());
            
            glPushMatrix();
            glMultMatrixf(elem->matrix().m);
            elem->mesh()->renderWithoutMaterial();
            glPopMatrix();
        }
        
        glDepthMask(GL_TRUE);    
        m_shadowPassShader.disable();        
        
        stateTransp_3.pop();
        renderNoMateState_2.pop();                
    }

    stateDisMat_1.pop();        
    glPopAttrib(); // #glPushAttrib(GL_POLYGON_BIT);

    Matrix44f projMatrix;
    projection.genMatrix(&projMatrix);
    for (Scene::WaterPlanesList::iterator it = scene.waterPlanes.begin();
         it != scene.waterPlanes.end(); it++)
    {
        (*it)->draw(scene, matView, projMatrix);
    }

    for (Scene::SkyboxList::iterator it = scene.skyboxes.begin();
         it != scene.skyboxes.end(); it++)
    {
        (*it)->draw();
    }
    
    //depthmap::ShowDepthMap(m_depthMap.oglName(), 0.0f, 0.0f, 1.0f, -1.0f);
}

UD_NAMESPACE_END

