#ifndef UD_PATHANIMATION_HPP
#define UD_PATHANIMATION_HPP

#include <vector>
#include <list>
#include "../common.hpp"
#include "../math/math.hpp"

UD_NAMESPACE_BEGIN

class Camera;
class World;

namespace pathanimation
{        
    /**
     * Linear path animation. Just linear interpolate the path points.
     */
    class Linear
    {
    public:
        /**
         * Constructor.
         * @param direction the model initial direction
         * @param locSec the translation ratio per seconds
         */
        Linear(const ud::Vec3f &direction,
               float locSec = 1.0f);
        
        /**
         * Clears the path and sets the instance to create a new one.
         * This method should be always called first.
         * @param startPoint the first point in the path
         */
        virtual void startPath(const Vec3f &startPoint);        
        
        /**
         * Adds a new point at end to the path.
         * Note: this method can't be called for the first point.
         * @param dest the new point
         */
        virtual void addPoint(const Vec3f &dest);                                                    
        
        /**
         * Updates the animation (i.e. makes the object move).
         * This method will update the instance absolute matrix.
         * Also to the animation work, is need to call every frame.
         * @param initialmatrix the initial transformation
         * @return true if the absolute matrix has changed.
         */
        bool update(float fps, const Matrix44f &initialmatrix = Matrix44f::Identity());

        /**
         * Returns the current point index.
         */
        int cpoint() const
        {
            return static_cast<int>(m_cpoint);
        }
        
        /**
         * Sets current point index.
         */
        void cpoint(int p)
        {
            m_cpoint = static_cast<float>(p);
        }

        /**
         * Returns the last absolute transformation. 
         * This matrix can be use as current transformation for 
         * a moving model or other node type.
         * This matrix is updated ways that the method update is
         * called and returned true.
         * @return the animation last transformation.
         */
        const Matrix44f& absolute() const
        {
            return m_absolute;
        }
        
        /**
         * Returns the distance that is travelled per second.
         */
        float locSec() const
        {
            return m_locSec;
        }
        
        /**
         * Sets the distance that is travelled per second.
         */
        void locSec(float v)
        {
            m_locSec = v;
        }

        /**
         * Returns the object original direction.
         */
        const Vec3f& direction() const
        {
            return m_direc;
        }
        
        /**
         * Sets the object original direction. This also sets the current
         * direction.
         */
        void direction(const Vec3f &direc)
        {
            m_direc = direc;
            m_cdirec = direc;
        }                

        /**
         * Returns the object current direction.
         */
        const Vec3f& cdirection() const
        {
            return m_cdirec;
        }
                
        /**
         * Returns the object current point.
         */
        const Vec3f& loc() const
        {
            return m_loc;
        }
                
        /**
         * Returns the object current axis.
         */
        const Vec3f& axis() const
        { 
            return m_axis;
        }

        /**
         * Sets the object axis.
         */
        void axis(const Vec3f &v)
        {
            m_axis = v;
        }
        
        /**
         * Returns the number of interpolation points.
         */
        int pointCount() const
        {
            return m_points.size();
        }
        
        /**
         * Returns the nth interpolation point.
         */
        const ud::Vec3f& point(int i) const
        {
            return m_points[i];
        }
                
    private:        
        void move(const Vec3f &V, const Vec3f &D, const Matrix44f &initialmatrix);

        std::vector<Vec3f> m_points;
        Matrix44f  m_absolute;        
        float m_locSec, m_cpoint;
        
    protected:
        Vec3f m_cdirec, m_direc,
            m_loc, m_axis;
    };
    
    /**
     * Catmull-Rom curve path animation. This implementation converts the curve points to
     * a Linear animation under a given precision.
     */
    class CatmullRom: public Linear
    {
    public:
        /**
         * Constructor.
         * @param cdirection the object direction
         * @param precision the number of points that each curve segment will be broken in linear paths.
         * @param locSec the distance that is travelled per second
         */
        CatmullRom(const ud::Vec3f &cdirection, int precision = 32,
                   float locSec = 1.0f);

        /**
         * Clears the path and sets the instance to create a new one.
         * This method should be always called first.
         * @param startPoint the first point in the path
         */
        void startPath(const Vec3f &startPoint);
        
        /**
         * Adds an new curve control point.
         */
        void addPoint(const Vec3f &dest);
        
        /**
         * Ends the curve segment and converts the control point in
         * linear paths segments. This method should be called to close
         * the curve.
         */
        void endPath();
        
        /**
         * Draws the curve. glColor3f affects the line color.
         */
        void draw();
        
        /**
         * Sets the curve precision - the number of linear points that the path is split. 
         * This method has effect in the next endPath call.
         */
        void precision(int value) 
        {
            m_precision = value;
        }
        
        /**
         * Returns the curve precision - the number of linear points that the path is split. 
         */
        int precision() const
        {
            return m_precision;
        }        
        
    private:
        std::list<Vec3f> m_ctrlpoints;
        int m_precision;
    };
}

UD_NAMESPACE_END

#endif
