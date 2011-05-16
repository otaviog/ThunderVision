#include <iostream>
#include "../debug.hpp"
#include "../math/math.hpp"
#include "pathanimation.hpp"

UD_NAMESPACE_BEGIN

namespace pathanimation
{
    Linear::Linear(const Vec3f &cdirection, float locSec)
        : m_cdirec(cdirection)
    {        
        m_locSec = locSec;
        cpoint(0);
    }

    inline float frac(float n)
    {
        static float dummy;
        return modff(n, &dummy);
    }

    bool Linear::update(float fps, const Matrix44f &initialmatrix)
    {        
        const float speed = m_locSec/fps;
        
        bool findmove = true;
        bool movefound = false;        
        
        do 
        {
            const int cipoint = static_cast<int>(m_cpoint);
            if ( cipoint < static_cast<int>(m_points.size()) - 1 )
            {                                    
                Vec3f V;
                
                const Vec3f cpoint = m_points[cipoint];
                const Vec3f npoint = m_points[cipoint + 1];
                
                const Vec3f cn = npoint - cpoint;            
                const float cndist = vecLength(cn);
                const float tincr = frac(m_cpoint) + speed/cndist;
                
                if ( cndist > speed && tincr < 1.0f ) 
                {
                    V = cpoint*(1.0f - tincr) + npoint*tincr;
                    move(V, vecNormal(cn), initialmatrix);
                    findmove = false;
                    movefound = true;
                }
                m_cpoint += speed/cndist;
            }
            else
            {
                findmove = false;
            }
        } while(findmove);
                                     
        return movefound;
    }

    void Linear::move(const Vec3f &V, const Vec3f &D, const Matrix44f &initialmatrix)
    {
        m_loc = V;
        m_absolute = ud::Matrix44f::Translation(V);        
        
        const float cos = vecDot(m_direc, D);
        float rotangle;
        
        if ( fltLt(cos, 1.0f, 5.0e-6f) )
        {
            rotangle = 0.0f;
        
            if ( fltLt(cos, -1.0f, 5.0e-6f) )
            {                
                rotangle = std::acos(-1.0f);
            }
            else
            {
                m_axis = vecNormal(vecCross(m_direc, D));
                rotangle = std::acos(cos);
            }
            
            m_absolute = m_absolute*Matrix44f(rotationQuat(rotangle, m_axis));            
            m_cdirec = D;
        }         
    }
    
    void Linear::addPoint(const Vec3f &dest)
    {
        if ( (m_points.size() > 0 && vecLength(dest - m_points.back()) > UD_EPSILON)
             || m_points.size() == 0 )
        {
            m_points.push_back(dest);            
        }
    }    

    void Linear::startPath(const Vec3f &startPoint)
    {
        if ( m_points.empty() )
        {
            m_loc = startPoint;
            m_points.push_back(startPoint);
        }
        else
        {    
            m_points.erase(m_points.begin(), m_points.end());
            if ( !((m_loc - startPoint) == Vec3f(0.0f)) ) 
            {
                m_points.push_back(m_loc);
                m_points.push_back(startPoint);
            }
            else
            {
                m_points.push_back(startPoint);
            }
        }        
        
        cpoint(0);
    }

    CatmullRom::CatmullRom(const ud::Vec3f &cdirection, int precision, float locSec)
        : Linear(cdirection, locSec)
    {
        m_precision = precision;
    }

    void CatmullRom::startPath(const Vec3f &startPoint)
    {
        m_ctrlpoints.clear();
        m_ctrlpoints.push_back(startPoint);
    }
    
    void CatmullRom::addPoint(const Vec3f &dest)
    {
        m_ctrlpoints.push_back(dest);
    }
    
    static Vec3f catmulRom(const Vec3f &p1, const Vec3f &p2, const Vec3f &p3,
                           const Vec3f &p4, float t)
    {        
        return ((-p1 + 3.0f*p2 - 3.0f*p3 + p4)*t*t*t
                + (2.0f*p1 - 5.0f*p2 + 4.0f*p3 - p4)*t*t
                + (-p1 + p3)*t + 2*p2)*0.5f;
    }
       
    
    void CatmullRom::endPath()
    {
        bool first = true;        
        std::list<Vec3f>::const_iterator startPoint = m_ctrlpoints.begin();        

        for(int ccpts = 0; m_ctrlpoints.size() - ccpts > 3; ++ccpts, ++startPoint)
        {
            std::list<Vec3f>::const_iterator b = startPoint;         
            const Vec3f &p1 = *b++;
            const Vec3f &p2 = *b++;
            const Vec3f &p3 = *b++;
            const Vec3f &p4 = *b;
            
            for (int i = 0; i<=m_precision; i++)
            {
                const float t = float(i)/float(m_precision);
                const Vec3f point(catmulRom(p1, p2, p3, p4, t));
            
                if ( first )
                {
                    Linear::startPath(point);
                    first = false;                
                }
                else
                {
                    Linear::addPoint(point);                
                }            
            }           
        }
    }

    void CatmullRom::draw()
    {       
        //static ud::StaticModel *sphere = 
        //ud::quadric::Sphere(0.5f, ud::Material(ud::Blue*0.5f, ud::Blue*0.5f, ud::White*0.4f, 127.0f), 16, 16);
        
        std::list<Vec3f>::const_iterator base = m_ctrlpoints.begin();        
        
        for(int ccpts = 0; m_ctrlpoints.size() - ccpts > 3; ++ccpts, ++base)
        {
            std::list<Vec3f>::const_iterator b = base;         
            const Vec3f &p1 = *b++;
            const Vec3f &p2 = *b++;
            const Vec3f &p3 = *b++;
            const Vec3f &p4 = *b;
#if 0            
            //ud::VBOMesh::GLState rstate;
            rstate.push();
            
            glPushMatrix();
            glTranslatef(p1[0], p1[1], p1[2]);
            //sphere->mesh()->render();
            glPopMatrix();
            
            glPushMatrix();
            glTranslatef(p2[0], p2[1], p2[2]);
            //sphere->mesh()->render();
            glPopMatrix();

            glPushMatrix();
            glTranslatef(p3[0], p3[1], p3[2]);
            //sphere->mesh()->render();
            glPopMatrix();
            
            glPushMatrix();
            glTranslatef(p4[0], p4[1], p4[2]);
            //sphere->mesh()->render();
            glPopMatrix();

            rstate.pop();
#endif       
            glPushAttrib(GL_LIGHTING_BIT | GL_POLYGON_BIT);
            glDisable(GL_LIGHTING);
            glBegin(GL_LINE_STRIP);        
            
            for (int i = 0; i<=m_precision; i++)
            {
                glVertex3fv(
                    catmulRom(p1, p2, p3, p4, float(i)/float(m_precision)).v);
            }
            glEnd();
            glPopAttrib();
        }// for
        
        glPushAttrib(GL_LIGHTING_BIT | GL_POLYGON_BIT);
        glDisable(GL_LIGHTING);
        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3fv(m_loc.v);
        glVertex3fv((m_loc + m_cdirec).v);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3fv(m_loc.v);
        glVertex3fv((m_loc + m_axis).v);

        glEnd();
        glPopAttrib();
    }   
}

UD_NAMESPACE_END
