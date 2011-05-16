#include "../math/vec3.hpp"
#include "camera.hpp"
#include "projection.hpp"
#include "stereo.hpp"

UD_NAMESPACE_BEGIN

Stereo::Stereo()
{
	m_aperture = 45.0f;
	m_focallength = 2.5f;
	m_eyeseparation = 0.003f;
}

Stereo::Stereo(float aperture, float focallength, float eyeseparation)
{
	m_aperture = aperture;
	m_focallength = focallength;
	m_eyeseparation = eyeseparation;
}

AnaglyphStereo::AnaglyphStereo()
{
}

AnaglyphStereo::AnaglyphStereo(float aperture, float focallength, float eyeseparation)
	: Stereo(aperture, focallength, eyeseparation)
{
}

void AnaglyphStereo::enable()
{
}

void AnaglyphStereo::applyGL(StereoEye eye, const Camera &camera, const Projection &proj)
{
	const Vec3f view(camera.getView() - camera.getEye());
	const Vec3f eyeDisp(vecNormal(vecCross(view, camera.getUp()))*(m_eyeseparation*0.5f));

	Vec3f camEye;
	if ( eye == Eye1 )
	{
		glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
		camEye = camera.getEye() + eyeDisp;
	}
	else
	{
		glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_FALSE);
		camEye = camera.getEye() - eyeDisp;
	}

	//glDrawBuffer(GL_BACK);
	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	proj.applyGL();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	const Vec3f camView = camEye + view;
	gluLookAt(camEye[0], camEye[1], camEye[2], 
		camView[0], camView[1], camView[2],
		camera.getUp()[0], camera.getUp()[1], camera.getUp()[2]);
}

void AnaglyphStereo::disable()
{
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
}

SplitStereo::SplitStereo(Split spl)
	: m_split(spl)
{
}

SplitStereo::SplitStereo(Split spl, float aperture, float focallength, float eyeseparation)
	: Stereo(aperture, focallength, eyeseparation), m_split(spl)
{
}

void SplitStereo::enable()
{
	glGetIntegerv(GL_VIEWPORT, m_savedviewport);
}

/**
 * Taken from  Virtual Reality Interactive Environment by Marcio S. Pinho, Mauro C. C. dos Santos and Regis A. P. Kopper.
 */
void SplitStereo::applyGL(StereoEye eye, const Camera &camera, const Projection &proj)
{
	const Vec3f view(camera.getView() - camera.getEye());
	const Vec3f eyeDisp(vecNormal(vecCross(view, camera.getUp()))*(m_eyeseparation*0.5f));

	Vec3f camEye;
	float left, right, bottom, top;

	const int width = m_savedviewport[2];
	const int height = m_savedviewport[3];

	float ratio;
	if ( height > width)
	{
		ratio = (float(width)/float(height))*0.5f;
	}
	else
	{
		ratio = (float(height)/float(width))*0.5f;
	}
	
	static const float DTOR = 0.0174532925f; //PI/180.0	
	const float radians = DTOR*m_aperture*0.5f;
	const float wd2 = radians*tan(radians);
	const float ndfl = proj.nearD()/m_focallength;

	if ( eye == Eye1 )
	{	
		if ( m_split == Horizontal )
		{
			glViewport(0, 0, width/2, height);
		}
		else
		{
			glViewport(0, 0, width, height/2);
		}

		camEye = camera.getEye() + eyeDisp;

		left = -ratio*wd2 + 0.5f*m_eyeseparation*ndfl;
		right = ratio*wd2 + 0.5f*m_eyeseparation*ndfl;
		bottom = -wd2;
		top = wd2;
	}
	else
	{	
		if ( m_split == Horizontal )
		{
			glViewport(width/2, 0, width, height);
		}
		else
		{
			glViewport(0, height/2, width, height);
		}

		camEye = camera.getEye() - eyeDisp;
		
		left = -ratio*wd2 - 0.5f*m_eyeseparation*ndfl;
		right = ratio*wd2 - 0.5f*m_eyeseparation*ndfl;
		bottom = -wd2;
		top = wd2;
	}
	
	glDrawBuffer(GL_BACK);	
	Projection(left, right, bottom, top, proj.nearD(), proj.farD()).applyGL();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	const Vec3f camView = camEye + view;
	gluLookAt(camEye[0], camEye[1], camEye[2], 
		camView[0], camView[1], camView[2],
		camera.getUp()[0], camera.getUp()[1], camera.getUp()[2]);
}

void SplitStereo::disable()
{
	glViewport(m_savedviewport[0], m_savedviewport[1], m_savedviewport[2], m_savedviewport[3]);
}

UD_NAMESPACE_END