#include "tripstripmesh.hpp"

TDV_NAMESPACE_BEGIN

TripStripMesh::TripStripMesh()
{
    m_nIndices = m_nVerts = 0;
}

void TripStripMesh::draw()
{
    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertices.get());
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_colors.get());
    glColorPointer(3, GL_FLOAT, 0, NULL);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indices.get());
    glDrawElements(GL_TRIANGLE_STRIP);
    
    glPopAttrib();
}

TDV_NAMESPACE_END
