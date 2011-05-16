#ifndef UD_CAMERA_HPP
#define UD_CAMERA_HPP

#include "../common.hpp"
#include "../math/math.hpp"

UD_NAMESPACE_BEGIN

class Aabb;

/**
 * Represents a camera in gluLookAt like format. 
 * This class use two points: eye, view and vector up.
 */
class Camera
{
public:
    /**
     * Constructs a camera with eye at 0.0f, view at 0.0f, 0.0f, -1.0f and up 0.0f, 1.0f, 0.0f.
     */
    Camera();
    
    /**
     * @param pos eye position
     * @param view view position
     * @param up up vector
     */
    Camera(const Vec3f &pos, const Vec3f &view,
           const Vec3f &up);

    /**
     *
     */
    Camera(float pos_x, float pos_y, float pos_z,
           float view_x, float view_y, float view_z,
           float up_x, float up_y, float up_z);

    /**
     * Rotates the view point around X axis.
     * @param angle angle in radians
     */
    void rotateX(float angle);
    
    /**
     * Rotates the view point around Y axis.
     * @param angle angle in radians
     */
    void rotateY(float angle);
    
    /**
     * Rotates the view point around Z axis.
     * @param angle angle in radians
     */
    void rotateZ(float angle);
    
    /**
     * Rotates the eye position around X axis.
     * @param angle angle in radians
     */
    void rotateEyeX(float angle);
    
    /**
     * Rotates the eye position around Y axis.
     * @param angle angle in radians
     */
    void rotateEyeY(float angle);
   
    /**
     * Set the camera view rotation by a quaternion
     * @param rot rotation quaternion
     */
    void setRotation(const Quatf &rot);

    void rotate(float xangle, float yangle);
    
    /**
     * Translate eye and view.
     * @param xt x
     * @param yt y
     * @param zt z
     */
    void translate(float xt, float yt, float zt);
    
    /**
     * Moves the eye and view. The direction is defined by view - eye.
     * @param ac amount of displacement     
     */
    void move(float ac);
    
    /**
     * Moves the eye and view by some vector.
     * @param move the move direction
     */
    void move(const Vec3f &move);
    
    void moveXZ(float ac);
    
    /**
     * Set camera eye conservating the view direction
     * @param pos new eye pos
     */
    void setPosition(const Vec3f &pos);
    
    /**
     * Strifes to relative left or right position (use right vector - cross(up, view - eye))
     * @param amount of displacement, minus means to left
     */
    void strife(float ac);

    /**
     * Generates a matrix with camera transformations.
     * @param matrix return matrix
     */
    void genMatrix(Matrix44f *matrix) const;
    
    /**
     * Set camera transformation to current OpenGL state. (calls glLoadIdentity() first)
     */
    void applyGL() const;
    
    /**
     * Apply camera transformation to current OpenGL state.
     */
	void lookAt() const;
    
    void setAabb(const Aabb &box, const Vec3f &eyeVec, float distance);
    
    /**
     * Sets camera values.
     *
     */
	void set(float ex, float ey, float ez,
			 float vx, float vy, float vz,
			 float ux, float uy, float uz)
	{
		setEye(ex, ey, ez);
		setView(vx, vy, vz);
		setUp(ux, uy, uz);
	}

    /**
     * Sets camera values.
     *
     */
	void set(Vec3f eye, Vec3f view, Vec3f up)
	{
		setEye(eye);
		setView(view);
		setUp(up);
	}
    
    /**
     * Sets eye position
     */
    void setEye(float x, float y, float z)
    {
        setEye(Vec3f(x, y, z));
    }
    
    /**
     * Sets eye position
     */
    void setEye(const Vec3f &eye)
    {
        m_eye = eye;
    }

    /**
     * Sets view position.
     */
    void setView(float x, float y, float z)
    {
        setView(Vec3f(x, y, z));
    }
    
    /**
     * Sets view position.
     */
    void setView(const Vec3f &view)
    {
        m_view = view;
    }

    /**
     * Sets up vector.
     */
    void setUp(float x, float y, float z)
    {
        setUp(Vec3f(x, y, z));
    }
    
    /**
     * Sets up vector.
     */
    void setUp(const Vec3f &up)
    {
        m_up = up;
    }

    /**
     * Returns eye position.
     */
    const Vec3f& getEye() const
    {
        return m_eye;
    }

    /**
     * Returns view position.
     */
    const Vec3f& getView() const
    {
        return m_view;
    }

    /**
     * Returns up vector.
     */
    const Vec3f& getUp() const
    {
        return m_up;
    }

    /**
     * Returns normalied camera direction vector.
     */
    Vec3f getDirection() const
    { return vecNormal(m_view - m_eye); }
        
private:
    Vec3f m_eye, m_view, m_up;
};

UD_NAMESPACE_END

#endif
